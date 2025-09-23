# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from dataset import DatasetFromDirectory

class CNN1DModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(dropout_rate)
        self.fc1   = nn.Linear(128 * 8, 256)  # 68→34→17→8 after three pools
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = nn.ReLU()(self.bn1(self.conv1(x))); x = self.pool(x)
        x = nn.ReLU()(self.bn2(self.conv2(x))); x = self.pool(x)
        x = nn.ReLU()(self.bn3(self.conv3(x))); x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def fit_model(model, train_loader, val_loader, optimizer, criterion,
              max_epochs, device, model_name):
    model.to(device)
    loss_history = []

    for epoch in tqdm(range(1, max_epochs+1), desc=f"Epochs ({model_name})", unit="epoch"):
        model.train()
        batch_losses = []
        for Xb, yb in tqdm(train_loader, desc="Batches", leave=False, unit="batch"):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        loss_history.append(np.mean(batch_losses))

        if epoch % 50 == 0 or epoch == max_epochs:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device)
                    p = model(Xb).argmax(dim=1).cpu().numpy()
                    preds.append(p)
                    trues.append(yb.numpy())
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(trues)
            print(f"\n[{model_name}] Epoch {epoch} | Val Acc: {accuracy_score(y_true, y_pred):.4f}")
            print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
            print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

    return loss_history

def main():
    parser = argparse.ArgumentParser(description="Train CNN1D with hold‑out + k‑fold")
    parser.add_argument('--data_root',    type=str, default='/media/sma318/TOSHIBA EXT/MobiLyzer/datasets/classification_resubmit/milk')
    parser.add_argument('--fruit',        type=str, default='milk')
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--n_splits',     type=int,   default=4)
    parser.add_argument('--max_epochs',   type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--random_state', type=int,   default=0)
    args = parser.parse_args()

    # — sanity check —
    ds = DatasetFromDirectory(args.data_root, "", args.fruit)
    print(f"→ Found {len(ds)} samples in '{args.data_root}' (fruit='{args.fruit}')")
    if len(ds) == 0:
        raise RuntimeError("No data found! Check your data_root and sub‑folders.")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    X_list, y_list = [], []
    for sig, label in loader:
        X_list.append(sig.squeeze().numpy())
        y_list.append(label.item())
    X = np.stack(X_list, 0)
    y = np.array(y_list)
    print("Data loaded:", X.shape, y.shape)

    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=args.test_size,
        stratify=y, random_state=args.random_state)
    os.makedirs("Models", exist_ok=True)
    np.save("Models/X_test.npy", X_test)  # note: you probably meant to save y_test here
    np.save("Models/y_test.npy", y_test)

    scaler = MinMaxScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trval, y_trval)):
        print(f"\n=== Fold {fold} ===")
        X_train, y_train = X_trval[train_idx], y_trval[train_idx]
        X_val,   y_val   = X_trval[val_idx],   y_trval[val_idx]

        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        train_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        ), batch_size=args.batch_size, shuffle=True)

        val_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        ), batch_size=args.batch_size, shuffle=False)

        model     = CNN1DModel(input_channels=1, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cw        = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))

        history = fit_model(model, train_loader, val_loader,
                            optimizer, criterion,
                            args.max_epochs, device,
                            f"CNN1D_fold{fold}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'loss_history': history
        }, f"Models/CNN1D_{args.fruit}_fold{fold}.pt")
        print(f"Saved Models/CNN1D_{args.fruit}_fold{fold}.pt")

if __name__ == "__main__":
    main()
