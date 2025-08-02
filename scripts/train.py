import os
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
import multiprocessing
import joblib

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.CheekRPPGDataset import CheekRPPGDataset
from models.CNNLSTMModel import CNNLSTMModel
from utils.model_summary import summarize_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sequence_length", type=int, default=25)
parser.add_argument("--stride", type=int, default=25)
parser.add_argument("--target", type=str, default="SBP")
parser.add_argument("--use_rppg", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--rppg_mode", type=str, default="frame")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--run_id", type=str, default="default")
args = parser.parse_args()

# Parameters
train_data_root = "./data/processed/train/"
val_data_root = "./data/processed/val/"
sequence_length = args.sequence_length
stride = args.stride
target = args.target
use_rppg = args.use_rppg
rppg_mode = args.rppg_mode # "sequence" or "frame"
batch_size = args.batch_size
accumulation_steps = 4 # Gradient accumulation steps
num_epochs = args.epochs
lr = args.lr
lstm_hidden_dim = 32
model_save_path = f"./models/cnn_lstm_model{args.run_id}"
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# print(f"run_id: {args.run_id}")

# Get all subject folders
train_subjects = sorted([d for d in os.listdir(train_data_root) if os.path.isdir(os.path.join(train_data_root, d))])
val_subjects = sorted([d for d in os.listdir(val_data_root) if os.path.isdir(os.path.join(val_data_root, d))])

def get_dataloader(subjects, data_root, scaler=None):
    dataset = CheekRPPGDataset(
        data_root=data_root,
        split_list=subjects,
        sequence_length=sequence_length,
        stride=stride,
        target=target,
        use_rppg=use_rppg,
        rppg_mode=rppg_mode,
        transform=None,
        scaler=scaler
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True, pin_memory=True)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Create MinMaxScaler for scaling ground truths to [0,1]
    temp_dataset = CheekRPPGDataset(
        data_root = train_data_root,
        split_list=train_subjects,
        sequence_length=sequence_length,
        stride=stride,
        target=target,
        use_rppg=use_rppg,
        rppg_mode=rppg_mode
    )
    all_labels = np.array([entry["label"] for entry in temp_dataset.samples]).reshape(-1,1)
    label_scaler = MinMaxScaler()
    label_scaler.fit(all_labels)

    # Save scaler for future use
    joblib.dump(label_scaler, os.path.join(results_dir, f"scaler_{args.run_id}.pkl"))

    train_loader = get_dataloader(train_subjects, train_data_root, scaler=label_scaler)
    val_loader = get_dataloader(val_subjects, val_data_root, scaler=label_scaler)

    # Model
    # CHANGES MADE ON 14072025
    # model = CNNLSTMModel(lstm_hidden_dim=lstm_hidden_dim, use_rppg=use_rppg, freeze_ratio=0.9).to(device)
    model = CNNLSTMModel(lstm_hidden_dim=lstm_hidden_dim, use_rppg=use_rppg, freeze_ratio=0.9)
    model.load_state_dict(torch.load(f"{model_save_path}.pth", map_location=device))
    model.to(device)
    summarize_model(model, show_submodules=False)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Metrics storage
    metrics_log = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": []
    }

    # Mixed precision scaler
    grad_scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    best_val_mae = float('inf')
    best_epoch = None
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            left = batch["left_cheek"].to(device, non_blocking=True)
            right = batch["right_cheek"].to(device, non_blocking=True)
            rppg = batch["rppg"].to(device, non_blocking=True).float() if use_rppg else None
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(left, right, rppg)
                loss = criterion(outputs, labels) / accumulation_steps
            
            grad_scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
        
        train_mae = mean_absolute_error(train_labels, train_preds)
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                left = batch["left_cheek"].to(device, non_blocking=True)
                right = batch["right_cheek"].to(device, non_blocking=True)
                rppg = batch["rppg"].to(device, non_blocking=True).float() if use_rppg else None
                labels = batch["label"].to(device, non_blocking=True)

                # optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(left, right, rppg)
                    loss = criterion(outputs, labels) / accumulation_steps

                val_loss_total += loss.item() * accumulation_steps

                val_preds.extend(outputs.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
                
        val_preds_np = np.array(val_preds)
        val_labels_np = np.array(val_labels)

        if np.isnan(val_preds_np).any() or np.isnan(val_labels_np).any():
            print("NaNs detected in validation predictions or labels.")
            mask = ~np.isnan(val_preds_np) & ~np.isnan(val_labels_np)
            val_mae = mean_absolute_error(val_labels_np[mask], val_preds_np[mask])
        else:
            val_mae = mean_absolute_error(val_labels_np, val_preds_np)

        val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        metrics_log["epoch"].append(epoch + 1)
        metrics_log["train_loss"].append(train_loss)
        metrics_log["train_mae"].append(train_mae)
        metrics_log["val_loss"].append(val_loss)
        metrics_log["val_mae"].append(val_mae)

        # Save model every 10 epochs during the first 50 epochs in case of crash or something
        if (epoch + 1) <= 50 and (epoch + 1) % 10 == 0:
            best_epoch = epoch+1
            torch.save(model.state_dict(), f"{model_save_path}_ep{best_epoch}.pth")
            print(f"Checkpoint saved at epoch {best_epoch}")

        # # Early stopping check after 50 epochs
        if (epoch + 1) > 0:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), f"{model_save_path}_ep{epoch+51}.pth")  # Save best model
                print(f"New best model saved with Val MAE: {val_mae:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement in Val MAE for {patience_counter} epoch(s).")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Final save
    torch.save(model.state_dict(), f"{model_save_path}.pth")
    print(f"Model saved to {model_save_path}")

    # Save metrics
    metrics_path = os.path.join(results_dir, f"metrics_{args.run_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    # Loading the best model state for predictions
    if best_epoch is not None:
        model.load_state_dict(torch.load(f"{model_save_path}_ep{best_epoch}.pth"))

    # Save predictions
    def save_predictions(loader, split_name):
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predicting {split_name}"):
                left = batch["left_cheek"].to(device)
                right = batch["right_cheek"].to(device)
                rppg = batch["rppg"].to(device).float() if use_rppg else None
                labels = batch["label"].cpu().numpy()
                outputs = model(left, right, rppg).cpu().numpy()
                preds.extend(outputs.flatten().tolist())
                gts.extend(labels.flatten().tolist())
            preds = label_scaler.inverse_transform(np.array(preds).reshape(-1,1))
            gts = label_scaler.inverse_transform(np.array(gts).reshape(-1,1))
        with open(os.path.join(results_dir, f"{args.run_id}_{split_name}_predictions.json"), "w") as f:
            json.dump({"predictions": preds.tolist(), "ground_truth": gts.tolist()}, f)
        print(f"{split_name.capitalize()} predictions saved.")

    save_predictions(train_loader, "train")
    save_predictions(val_loader, "val")

    test_data_root = "./data/processed/test/"
    test_subjects = sorted([d for d in os.listdir(test_data_root) if os.path.isdir(os.path.join(test_data_root, d))])
    test_loader = get_dataloader(test_subjects, test_data_root, scaler=label_scaler)
    save_predictions(test_loader, "test")