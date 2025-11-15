import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import flwr as fl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", "--client-id", type=int, dest="cid", required=True)
    parser.add_argument("--csv", default="labels/ISIC_2020_Training_GroundTruth.csv")
    parser.add_argument("--img-dir", default="../train/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--server-address", default="localhost:8080")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class SmallCNN(nn.Module):
    # Smaller, efficient CNN with AdaptiveAvgPool to avoid huge FC
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x32x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256x16x16
        )

        # Adaptive pooling -> fixed small feature map size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> 256 x 4 x 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['image_name'] + ".jpg")
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = 0 if row['benign_malignant'] == 'benign' else 1
        return img, label


def load_partition(csv_path, hospital_id, img_dir, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    part_df = df[df['hospital_id'] == hospital_id]
    if part_df.empty:
        return None, None

    # stratified split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(part_df, test_size=test_size, stratify=part_df['benign_malignant'], random_state=random_state)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SkinLesionDataset(train_df, img_dir, transform=transform)
    val_ds = SkinLesionDataset(val_df, img_dir, transform=transform)
    return train_ds, val_ds


def train_local(model, dataloader, epochs, lr, device):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0

        # Add progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Local epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    return epoch_loss


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # tqdm evaluation bar
    pbar = tqdm(dataloader, desc="Evaluation", leave=False)

    with torch.no_grad():
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

            # update bar live
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, correct / total


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, test_ds, device, args):
        self.model = model
        self.trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.testloader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)
        self.device = device
        self.args = args

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {}
        for (key, _), param in zip(self.model.state_dict().items(), parameters):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_local(
            self.model,
            self.trainloader,
            epochs=config.get("local_epochs", 1),
            lr=config.get("lr", 1e-4),
            device=self.device
        )
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {args.cid} using device: {device}")

    # Load CSV and map cid -> hospital_id. We assume hospital_id values are known and stable.
    df = pd.read_csv(args.csv)
    hospital_ids = sorted(df['hospital_id'].unique())

    if args.cid >= len(hospital_ids):
        raise SystemExit(f"Client id {args.cid} out of range (found {len(hospital_ids)} hospital partitions).")

    my_hospital = hospital_ids[args.cid]
    print(f"Client {args.cid} will use hospital_id = {my_hospital}")

    train_ds, val_ds = load_partition(args.csv, my_hospital, args.img_dir)
    if train_ds is None or len(train_ds) == 0:
        raise SystemExit("No training data for this client")

    model = SmallCNN(num_classes=2)

    # Wrap client
    client = FlowerClient(model, train_ds, val_ds, device, args)

    # Start Flower client (this will connect to server)
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == '__main__':
    main()
