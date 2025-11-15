import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import flwr as fl
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import threading


class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input 3x256x256
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
            nn.MaxPool2d(2)   # 256x16x16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*16*16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class SkinLesionDataset(Dataset):
    def __init__(self, df, dir_path='../train/'):
        self.df = df.reset_index(drop=True)
        self.dir_path = dir_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.dir_path + row['image_name'] + ".jpg"
        img = Image.open(path).convert("RGB")
        img = np.array(img) / 255.0
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        label = 0 if row['benign_malignant'] == 'benign' else 1
        return img, label


def load_client_dataset(csv_path, hospital_id, dir_path='../train/'):
    df = pd.read_csv(csv_path)
    client_df = df[df["hospital_id"] == hospital_id]
    if client_df.empty:
        return None, None

    # Optional: split into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(client_df, test_size=0.2, random_state=42, stratify=client_df['benign_malignant'])

    train_dataset = SkinLesionDataset(train_df, dir_path)
    val_dataset = SkinLesionDataset(val_df, dir_path)
    return train_dataset, val_dataset


def train_fn(model, trainloader, epochs, lr, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return loss.item()

def test_fn(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_total += loss.item() * len(y_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return loss_total / total, correct / total


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, device):
        self.model = model
        self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=32)
        self.device = device

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_fn(self.model, self.trainloader, epochs=config["local_epochs"], lr=config["lr"], device=self.device)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test_fn(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def start_flower_server():
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
dir_path = '../train/'

df = pd.read_csv("labels/ISIC_2020_Training_GroundTruth.csv")
hospital_ids = df["hospital_id"].unique()

clients_train_data = []
clients_test_data = []

for hid in hospital_ids:
    train_ds, val_ds = load_client_dataset("labels/ISIC_2020_Training_GroundTruth.csv", hospital_id=hid)
    if train_ds is None:
        continue
    clients_train_data.append(train_ds)
    clients_test_data.append(val_ds)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=4,
    min_available_clients=4,
)

clients = []
for i in range(4):
    model = SmallCNN().to(device)
    clients.append(FlowerClient(model, clients_train_data[i], clients_test_data[i], device))

# --- Start the server in a background thread ---
server_thread = threading.Thread(target=start_flower_server, daemon=True)
server_thread.start()

# --- Start clients ---
def start_client(client_obj):
    fl.client.start_numpy_client("localhost:8080", client=client_obj)

client_threads = []
for client_obj in clients:  # your FlowerClient instances
    t = threading.Thread(target=start_client, args=(client_obj,))
    t.start()
    client_threads.append(t)

# Optional: wait for clients to finish
for t in client_threads:
    t.join()