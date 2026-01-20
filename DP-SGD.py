import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from opacus import PrivacyEngine
import pandas as pd
from opacus.validators import ModuleValidator
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from datetime import datetime
import sys
import warnings

# Suppress warnings and set logging
warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("opacus").setLevel(logging.ERROR)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
   import flwr as fl
except Exception:
    fl = None

try:
    import mlflow
except Exception:
    mlflow = None



def split_patients_federated(txt_path, num_clients=3, alpha=1.5, seed=54):
    """
    Splits data into non-IID client datasets using Dirichlet distribution.
    """
    np.random.seed(seed)

    df = pd.read_csv(
        txt_path, delim_whitespace=True, header=None, index_col=False,
        names=["patient_id", "filename", "label", "source"]
    )

    df["patient_id"] = df["filename"]
    client_assignments = {i: [] for i in range(num_clients)}

    for label_value in df['label'].unique():
        subset = df[df['label'] == label_value]
        patients = subset["patient_id"].unique()
        np.random.shuffle(patients)
        num_patients = len(patients)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        client_sizes = (proportions * num_patients).astype(int)
        
        while client_sizes.sum() < num_patients:
            client_sizes[np.argmax(proportions)] += 1

        idx = 0
        for cid, size in enumerate(client_sizes):
            assigned = patients[idx:idx + size]
            client_assignments[cid].extend(assigned)
            idx += size

    df["client_id"] = -1
    for cid in range(num_clients):
        df.loc[df["patient_id"].isin(client_assignments[cid]), "client_id"] = cid

    return df


def robust_read_txt(path):

    df_raw = pd.read_csv(path, delim_whitespace=True, header=None, dtype=str, keep_default_na=False)
    df_raw = df_raw.fillna("")
    cols = [f'c{i}' for i in range(df_raw.shape[1])]
    df_raw.columns = cols

    fname_col = None
    label_col = None
    for c in cols:
        s = df_raw[c].str.lower()
        if s.str.contains(r'\.(jpg|jpeg|png|bmp|gif)$').any():
            fname_col = c
        if s.isin(['positive', 'negative']).any():
            label_col = c

    if label_col is None:
        for c in cols:
            s = df_raw[c].str.lower()
            if s.str.contains('positive').any() or s.str.contains('negative').any():
                label_col = c
                break

    pid_col = None
    for c in cols:
        if c not in (fname_col, label_col):
            pid_col = c
            break

    source_col = None
    for c in cols:
        if c not in (fname_col, label_col, pid_col):
            source_col = c
            break

    out = pd.DataFrame()
    out['patient_id'] = df_raw[pid_col] if pid_col in df_raw else ""
    if fname_col and fname_col in df_raw:
        out['filename'] = df_raw[fname_col]
    else:
        # fallback to second column if available
        out['filename'] = df_raw[cols[1]] if len(cols) > 1 else df_raw[cols[0]]
    out['label'] = df_raw[label_col] if label_col and label_col in df_raw else ""
    out['source'] = df_raw[source_col] if source_col and source_col in df_raw else ""
    return out


def split_patients_federated_df(df, num_clients=3, alpha=1.5, seed=54):

    np.random.seed(seed)
    df = df.copy()
    df['patient_id'] = df['patient_id'].astype(str)
    client_assignments = {i: [] for i in range(num_clients)}

    for label_value in df['label'].unique():
        subset = df[df['label'] == label_value]
        patients = subset['patient_id'].unique()
        np.random.shuffle(patients)
        num_patients = len(patients)
        proportions = np.random.dirichlet([alpha] * num_clients)
        client_sizes = (proportions * num_patients).astype(int)
        while client_sizes.sum() < num_patients:
            client_sizes[np.argmax(proportions)] += 1
        idx = 0
        for cid, size in enumerate(client_sizes):
            assigned = patients[idx:idx + size]
            client_assignments[cid].extend(assigned)
            idx += size

    df['client_id'] = -1
    for cid in range(num_clients):
        df.loc[df['patient_id'].isin(client_assignments[cid]), 'client_id'] = cid
    return df


class CovidDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {"negative": 0, "positive": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        filename = None

        if 'filename' in row and str(row['filename']).lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = str(row['filename']).strip()
        else:

            for v in row.values:
                s = str(v).strip()
                if s.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    filename = s
                    break
        
        if filename is None:
            try:
                filename = str(row.iloc[1]) 
            except:
                raise RuntimeError(f"Could not find filename for row {idx}: {row.values}")

        img_path = os.path.join(self.root_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open {img_path} - {e}")
            image = Image.new('RGB', (224, 224))

        label = None
        if 'label' in row:
            s = str(row['label']).lower().strip()
            if s in self.label_map:
                label = self.label_map[s]
        
        if label is None:
            for v in row.values:
                s = str(v).lower().strip()
                if s in self.label_map:
                    label = self.label_map[s]
                    break
        
        if label is None:
            for v in row.values:
                s = str(v).lower()
                if "positive" in s:
                    label = 1
                    break
                if "negative" in s:
                    label = 0
                    break
        
        if label is None:
            raise RuntimeError(f"Unknown label for row {idx}. content: {row.values}")

        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_model(use_dp=True):
    model = None

    try:
        print("[DEBUG] Attempting to load ConvNeXt-Tiny...")

        try:
            model = models.convnext_tiny(weights='DEFAULT')
        except (TypeError, AttributeError):
            if hasattr(models, 'convnext_tiny'):
                model = models.convnext_tiny(pretrained=True)
            else:
                raise ImportError("ConvNeXt not found in torchvision.models")

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 2)
        nn.init.xavier_uniform_(model.classifier[2].weight)
        nn.init.zeros_(model.classifier[2].bias)
        print("[DEBUG] ConvNeXt-Tiny loaded successfully.")

    except Exception as e:
        print(f"[DEBUG] ConvNeXt load failed ({e}). Falling back to ResNet18.")
        try:
            model = models.resnet18(weights='DEFAULT')
        except:
            model = models.resnet18(pretrained=True)
            
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)

        if use_dp:
            for m in model.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False
            
            try:
                from torchvision.models.resnet import BasicBlock
                def _patched_basicblock_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = F.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    if self.downsample is not None:
                        identity = self.downsample(x)
                    out = out + identity
                    out = F.relu(out)
                    return out
                BasicBlock.forward = _patched_basicblock_forward
            except:
                pass
    if use_dp:

        from torchvision.ops import StochasticDepth
        def replace_stochastic_depth_with_identity(module):
            for name, child in module.named_children():
                if isinstance(child, StochasticDepth):
                    setattr(module, name, nn.Identity())
                else:
                    replace_stochastic_depth_with_identity(child)
        
        if model is not None:
            replace_stochastic_depth_with_identity(model)
            print("[DEBUG] Disabled StochasticDepth layers for Opacus compatibility.")

        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    return model


def local_train(model, train_loader, epochs, device, target_epsilon, target_delta, client_id=None):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    print(f"[DEBUG] Client {client_id}: Initializing PrivacyEngine...")
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=1.0,
        grad_sample_mode="hooks",
    )
    print(f"[DEBUG] Client {client_id}: PrivacyEngine attached.")

    weights = None
    try:
        ds = getattr(train_loader, 'dataset', None)
        if ds is not None and hasattr(ds, 'df'):
            labels = ds.df['label'].tolist()
            mapped = []
            for l in labels:
                try:
                    mapped.append(ds.label_map.get(str(l).lower(), int(l)))
                except Exception:
                    try:
                        mapped.append(int(l))
                    except Exception:
                        pass
            if len(mapped) > 0:
                from collections import Counter
                cnt = Counter(mapped)
                n_neg = cnt.get(0, 0)
                n_pos = cnt.get(1, 0)
                if n_neg > 0 and n_pos > 0:
                    total = float(n_neg + n_pos)
                    w_neg = total / (2.0 * n_neg)
                    w_pos = total / (2.0 * n_pos)
                    weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
    except Exception:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        correct = 0
        total_samples = 0
        prefix = f"Client {client_id}" if client_id is not None else "Global"
        loader_iter = tqdm(train_loader, desc=f"{prefix} Ep {epoch+1}/{epochs}", leave=False)
        for images, labels in loader_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if batch_count % 10 == 0:
                avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
                acc = correct / total_samples if total_samples > 0 else 0.0
                try:
                    loader_iter.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})
                except Exception:
                    pass

        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        train_acc = correct / total_samples if total_samples > 0 else 0.0
        prefix = f"Client {client_id}" if client_id is not None else "Global"
        print(f"{prefix} - Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - TargetEps: {target_epsilon}")

        if mlflow is not None:
            try:
                mlflow.log_metric(f"{prefix}_loss", float(avg_loss), step=epoch)
                mlflow.log_metric(f"{prefix}_acc", float(train_acc), step=epoch)
                mlflow.log_metric(f"{prefix}_epsilon", float(target_epsilon), step=epoch)
            except Exception:
                pass

            
    base_model = getattr(model, "_module", None) or getattr(model, "module", None) or model
    return base_model.state_dict()


def get_model_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_model_parameters(model, params):
    state_dict = model.state_dict()
    new_state = {}
    for (k, v), arr in zip(state_dict.items(), params):
        tensor = torch.tensor(arr, dtype=v.dtype)
        new_state[k] = tensor
    model.load_state_dict(new_state)


def federated_averaging(model, client_state_dicts):

    if not client_state_dicts:
        return model

    global_state = model.state_dict()
    num_clients = len(client_state_dicts)
    averaged_state = {}

    for name, param in global_state.items():
        accum = None
        for cs in client_state_dicts:
            if isinstance(cs, dict):
                v = cs.get(name)
            else:
                v = None

            if v is None:
                continue

            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).to(dtype=param.dtype)
            elif isinstance(v, torch.Tensor):
                t = v.to(dtype=param.dtype)
            else:
                try:
                    t = torch.tensor(v, dtype=param.dtype)
                except Exception:
                    continue

            if accum is None:
                accum = t.clone().detach().to(torch.float64)
            else:
                accum = accum + t.to(torch.float64)

        if accum is None:

            averaged_state[name] = param
        else:
            averaged = (accum / float(num_clients)).to(dtype=param.dtype)
            averaged_state[name] = averaged

    model.load_state_dict(averaged_state)
    return model

class FlowerClient(fl.client.NumPyClient if fl is not None else object):
    def __init__(self, cid, device):
        self.cid = cid
        self.device = device
        self.model = get_model(use_dp=True).to(self.device)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        try:
            set_model_parameters(self.model, parameters)
            cid_int = int(self.cid)
            client_df = train_df_federated[train_df_federated['client_id'] == cid_int]
            client_dataset = CovidDataset(client_df, IMG_DIR, transform=transform)
            client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

            weights = local_train(
                self.model, client_loader, LOCAL_EPOCHS, self.device, 
                EPSILON, DELTA, client_id=cid_int
            )
            try:
                client_model = get_model(use_dp=True).to(self.device)
                client_model.load_state_dict(weights)
                save_model(client_model, f"client{cid_int}_local")
                try:
                    evaluate_and_report(client_model, val_df, self.device, split_name=f"client{cid_int}_val", save_prefix=f"client{cid_int}_val", root_dir="val")
                except Exception as e:
                    print(f"Warning: Failed to run client validation save for client {cid_int} - {e}")
            except Exception as e:
                print(f"Warning: Failed to save client model for client {cid_int} - {e}")

            params = [v.cpu().numpy() for v in weights.values()]
            return params, len(client_dataset), {}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL CLIENT ERROR (Client {self.cid}): {e}")
            raise e

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        dataset = CovidDataset(val_df, VAL_DIR if 'VAL_DIR' in globals() else "val", transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                bs = labels.size(0)
                total_loss += loss.item() * bs
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += bs

        loss_val = float(total_loss / total) if total > 0 else 0.0
        accuracy = float(correct / total) if total > 0 else 0.0
        try:
            evaluate_and_report(self.model, val_df, self.device, split_name=f"client{self.cid}_evaluate", save_prefix=f"client{self.cid}_evaluate", root_dir="val")
        except Exception as e:
            print(f"Warning: Failed to save evaluation report for client {self.cid} - {e}")

        return loss_val, total, {"accuracy": accuracy}


NUM_CLIENTS = 3
ROUNDS = 14
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
EPSILON = 8.0
DELTA = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "train"
VAL_DIR = "val"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
TRAINED_DIR = os.path.join(ROOT_DIR, "trained_models")

QUICK_CHECK = False
QUICK_CHECK_EPOCHS = 3
QUICK_CHECK_MAX_SAMPLES = 512

if os.environ.get("DP_SGD_QUICK") == "1":
    print("[DEBUG] QUICK_CHECK enabled via DP_SGD_QUICK=1")
    QUICK_CHECK = True

if os.path.exists("train.txt"):
    try:
        train_raw = robust_read_txt("train.txt")
        train_df_federated = split_patients_federated_df(train_raw, num_clients=NUM_CLIENTS)
    except Exception as e:
        print(f"Warning: robust parsing train.txt failed - {e}, falling back to original reader")
        train_df_federated = split_patients_federated("train.txt", num_clients=NUM_CLIENTS)

    try:
        test_df = robust_read_txt("test.txt")
    except Exception:
        test_df = pd.read_csv("test.txt", delim_whitespace=True, header=None, names=["patient_id", "filename", "label", "source"], dtype=str, keep_default_na=False)

    try:
        val_df = robust_read_txt("val.txt")
    except Exception:
        val_df = pd.read_csv("val.txt", delim_whitespace=True, header=None, names=["patient_id", "filename", "label", "source"], dtype=str, keep_default_na=False)
else:
    print("WARNING: train.txt not found.")
    train_df_federated, test_df, val_df = None, None, None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_and_report(model, df, device, split_name="eval", save_prefix=None, root_dir=None):
    model.eval()
    data_root = root_dir or IMG_DIR
    dataset = CovidDataset(df, data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\n--- Evaluation ({split_name}) ---")
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
    print(report)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {split_name}')
    os.makedirs(TRAINED_DIR, exist_ok=True)
    if save_prefix is None: save_prefix = split_name
    cm_path = os.path.join(TRAINED_DIR, f"confmat_{save_prefix}.png")
    try:
        plt.savefig(cm_path, bbox_inches='tight')
        print(f"Saved confusion matrix: {cm_path}")
    except Exception as e:
        print(f"Warning: Failed to save confusion matrix {cm_path} - {e}")
    finally:
        plt.close()
    try:
        npy_path = os.path.join(TRAINED_DIR, f"confmat_{save_prefix}.npy")
        np.save(npy_path, cm)
        print(f"Saved confusion matrix array: {npy_path}")
    except Exception as e:
        print(f"Warning: Failed to save confusion matrix array {npy_path} - {e}")
    try:
        report_path = os.path.join(TRAINED_DIR, f"classification_report_{save_prefix}.txt")
        with open(report_path, 'w') as fh:
            fh.write(report)
        print(f"Saved classification report: {report_path}")
    except Exception as e:
        print(f"Warning: Failed to save classification report {report_path} - {e}")

    return cm

def quick_train_single_client(device, client_id=0, epochs=3, max_samples=None):
    print(f"\n=== Quick Check: Non-Private Training (Client {client_id}) ===")
    client_df = train_df_federated[train_df_federated['client_id'] == client_id]
    if max_samples:
        client_df = client_df.sample(n=min(len(client_df), max_samples), random_state=42).reset_index(drop=True)
    
    counts = client_df['label'].value_counts()
    n_neg = counts.get('negative', 0)
    n_pos = counts.get('positive', 0)
    if n_neg > 0 and n_pos > 0:
        total = n_neg + n_pos
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)
        weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
        print(f"Using Class Weights -> Neg: {w_neg:.2f}, Pos: {w_pos:.2f}")
    else:
        weights = None
        print("Warning: Missing a class, skipping weighted loss.")

    dataset = CovidDataset(client_df, IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = get_model(use_dp=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loader_iter = tqdm(loader, desc=f"Quick Client {client_id} Ep {ep+1}/{epochs}", leave=False)
        for images, labels in loader_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (total // BATCH_SIZE) % 10 == 0:
                avg_loss = running_loss / (total / BATCH_SIZE) if total > 0 else 0.0
                acc = correct / total if total > 0 else 0.0
                try:
                    loader_iter.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})
                except Exception:
                    pass
        avg_loss = running_loss / len(loader)
        acc = correct / total if total > 0 else 0
        print(f"QuickEpoch {ep+1}: Loss {avg_loss:.4f} | Acc {acc:.4f}")

        if mlflow is not None:
            try:
                mlflow.log_metric(f"quick_client{client_id}_loss", float(avg_loss), step=ep)
                mlflow.log_metric(f"quick_client{client_id}_acc", float(acc), step=ep)
            except Exception:
                pass
    evaluate_and_report(model, val_df, device, split_name="quick_val", save_prefix="quick_val", root_dir="val")


def save_model(model, prefix):

    os.makedirs(TRAINED_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    state_path = os.path.join(TRAINED_DIR, f"{prefix}_{ts}.pth")
    full_path = os.path.join(TRAINED_DIR, f"{prefix}_{ts}_full.pth")
    try:
        torch.save(model.state_dict(), state_path)
        print(f"Saved model state_dict: {state_path}")
    except Exception as e:
        print(f"Warning: Failed to save state_dict {state_path} - {e}")
    try:
        torch.save(model, full_path)
        print(f"Saved full model: {full_path}")
    except Exception as e:
        print(f"Warning: Failed to save full model {full_path} - {e}")
    # Log artifacts to MLflow if available
    if mlflow is not None:
        try:
            mlflow.log_artifact(state_path)
        except Exception:
            pass
        try:
            mlflow.log_artifact(full_path)
        except Exception:
            pass


def run_local_simulation():

    if train_df_federated is None:
        raise RuntimeError("No train data available to run local simulation.")

    print("[DEBUG] Starting local federated simulation loop...")
    global_model = get_model(use_dp=True).to(DEVICE)
    for r in range(ROUNDS):
        print(f"Local Sim - Round {r+1}/{ROUNDS}")
        client_weights = []
        for i in range(NUM_CLIENTS):
            client_df = train_df_federated[train_df_federated['client_id'] == i]
            client_dataset = CovidDataset(client_df, IMG_DIR, transform=transform)
            client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

            local_model = get_model(use_dp=True).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            weights = local_train(local_model, client_loader, LOCAL_EPOCHS, DEVICE, EPSILON, DELTA, client_id=i)
            client_weights.append(weights)

        global_model = federated_averaging(global_model, client_weights)

        save_model(global_model, f"global_model_local_round{r+1}")

        evaluate_and_report(global_model, val_df, DEVICE, split_name=f"val_local_round{r+1}", save_prefix=f"val_local_round{r+1}_{ts}", root_dir="val")

    save_model(global_model, "global_model_local_final")
    final_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    evaluate_and_report(global_model, test_df, DEVICE, split_name="test_local_final", save_prefix=f"test_local_final_{final_ts}", root_dir="test")


def client_fn(cid: str):
    return FlowerClient(cid, DEVICE)

def get_evaluate_fn(test_df, device):

    def evaluate(server_round, parameters, config):

        model = get_model(use_dp=True).to(device)
        set_model_parameters(model, parameters)
        
        cm = evaluate_and_report(
            model, 
            test_df, 
            device, 
            split_name=f"global_test_round_{server_round}", 
            save_prefix=f"global_round_{server_round}",
            root_dir="test"
        )
        
        save_model(model, f"global_model_round_{server_round}")
        
        return 0.0, {"accuracy": (cm.diagonal().sum() / cm.sum())}
    return evaluate

if __name__ == "__main__":

    run_mlflow = mlflow is not None
    if run_mlflow:
        try:
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "dp_sgd_experiment"))
        except Exception:
            pass
        try:
            run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_clients": NUM_CLIENTS,
                "rounds": ROUNDS,
                "local_epochs": LOCAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "epsilon": EPSILON,
                "delta": DELTA,
                "device": str(DEVICE),
            })
        except Exception:
            pass

    try:
        if QUICK_CHECK:
            quick_train_single_client(DEVICE, client_id=0, epochs=QUICK_CHECK_EPOCHS, max_samples=QUICK_CHECK_MAX_SAMPLES)
            print("\nQuick check complete.")
        else:
            if fl is None:
                # Fix for the missing 'ts' variable in run_local_simulation
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                print("Flower (flwr) is not installed — running local sequential simulation.")
                run_local_simulation()
            else:
                print(f"\nStarting Federated Learning Simulation using Flower...")

                strategy = fl.server.strategy.FedAvg(
                    fraction_fit=1.0,
                    fraction_evaluate=1.0,
                    min_fit_clients=NUM_CLIENTS,
                    min_evaluate_clients=NUM_CLIENTS,
                    min_available_clients=NUM_CLIENTS,
                    evaluate_fn=get_evaluate_fn(test_df, DEVICE) 
                )

                client_resources = {"num_cpus": 1, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0}

                fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=NUM_CLIENTS,
                    config=fl.server.ServerConfig(num_rounds=ROUNDS),
                    strategy=strategy,
                    client_resources=client_resources
                )
    finally:
        if run_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass