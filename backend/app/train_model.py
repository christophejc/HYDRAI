import os
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split

# Change the current working directory to the 'Project' folder in Google Drive
os.chdir('/content/drive/MyDrive/Project')
 
# Verify the current working directory
print(f"Current working directory: {os.getcwd()}")

from NormWear.main_model import NormWearModel
# ====================================================
# 1. Hydration Dataset (WINDOW + OVERLAP)
# ====================================================
LABEL_MAP = {
    "dehydrated_morning": 0,
    "hydrated_afternoon": 1,
    "mildly_hydrated_night": 2
}

class HydrationDataset(Dataset):
    def __init__(self, data_dir, window_size=150, step_size=50):
        self.samples = []
        global_index = 0

        for fname in os.listdir(data_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(data_dir, fname))

                X = df[["gsr_raw", "temp_raw", "hr_raw"]].values
                y = LABEL_MAP[str(df["label"].iloc[0]).strip().lower()]

                for start in range(0, len(X) - window_size, step_size):
                    window = X[start:start+window_size]

                    self.samples.append((
                        torch.tensor(window, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long),
                        global_index
                    ))
                    global_index += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ====================================================
# 2. Create Dataset + Stratified Split
# ====================================================
DATA_DIR = "/content/drive/MyDrive/Project/train_data"
dataset = HydrationDataset(DATA_DIR)

indices = list(range(len(dataset)))
labels  = [dataset[i][1].item() for i in indices]

# ---- (1) Train / Temp split ----
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
)

# ---- (2) Val / Test split ----
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=[labels[i] for i in temp_idx],
    random_state=42
)

# ---- (3) Create subsets ----
train_dataset = Subset(dataset, train_idx)
val_dataset   = Subset(dataset, val_idx)
test_dataset  = Subset(dataset, test_idx)

# ---- (4) DataLoaders ----
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ---- (5) Class distribution check ----
print("Train class count:", Counter([train_dataset[i][1].item() for i in range(len(train_dataset))]))
print("Val class count:",   Counter([val_dataset[i][1].item()   for i in range(len(val_dataset))]))
print("Test class count:",  Counter([test_dataset[i][1].item()  for i in range(len(test_dataset))]))

# ====================================================
# 3. NormWear Encoder -> Signal Embeddings
# ====================================================
device = torch.device("cpu")
encoder = NormWearModel(
    weight_path="normwear_last_checkpoint-15470-correct.pth",
    optimized_cwt=True
).to(device)

encoder.eval()
encoder.requires_grad_(False)

SAMPLING_RATE = 50

@torch.no_grad()
def extract_embeddings(loader):
    X_list, y_list, idx_list = [], [], []

    for xb, yb, idx in loader:
        # xb: [B, 150, 3] -> convert to [B, 3, 150]
        xb = xb.permute(0, 2, 1).contiguous().to(device)

        # embed
        embs = encoder.get_embedding(
            xb,
            sampling_rate=SAMPLING_RATE
        ).contiguous()

        # ------------------------------------------------------------
        # patch pooling and flatten
        # ----------------------------------------------------------
        if embs.dim() == 4:
            # [B, C, patches, 768]
            emb = embs.mean(dim=2)     # [B, C, 768]
            emb = emb.flatten(start_dim=1)  # [B, C*768]
        else:
            raise ValueError(f"Unexpected NormWear embedding shape: {embs.shape}")

        # store on CPU
        X_list.append(emb.cpu())
        y_list.append(yb.cpu())
        idx_list.append(idx.cpu())

    return (
        torch.cat(X_list),
        torch.cat(y_list),
        torch.cat(idx_list)
    )

# === EMBEDDINGS FOR ALL SPLITS ===
X_train, y_train, idx_train = extract_embeddings(train_loader)
X_val,  y_val,  idx_val     = extract_embeddings(val_loader)
X_test, y_test, idx_test    = extract_embeddings(test_loader)

torch.save({
    "X_train": X_train,
    "y_train": y_train,
    "idx_train": idx_train,

    "X_val": X_val,
    "y_val": y_val,
    "idx_val": idx_val,

    "X_test": X_test,
    "y_test": y_test,
    "idx_test": idx_test
}, "hydration_signal_embeddings.pt")

print("Saved raw NormWear embeddings to hydration_signal_embeddings.pt")

# ====================================================
# 4. Apple Watch features (steps + active calories)
# ====================================================

device = torch.device("cpu")
data = torch.load("hydration_signal_embeddings.pt")

X_train = data["X_train"]
y_train = data["y_train"]
idx_train = data["idx_train"]

X_val = data["X_val"]
y_val = data["y_val"]
idx_val = data["idx_val"]

X_test = data["X_test"]
y_test = data["y_test"]
idx_test = data["idx_test"]


files = [
    f"{DATA_DIR}/dehydrated.csv",
    f"{DATA_DIR}/hydrated.csv",
    f"{DATA_DIR}/mildly.csv"
]

APPLE_COLS = ["steps", "active_calories"]

WINDOW = 150
STEP   = 50                # overlap matches dataset

aw_all = []
for f in files:
    df = pd.read_csv(f)
    aw = df[APPLE_COLS].values
    for start in range(0, len(aw)-WINDOW, STEP):
        window = aw[start:start+WINDOW]
        aw_all.append(window.mean(axis=0))

aw_tensor = torch.tensor(np.array(aw_all, dtype=np.float32))


# ====================================================
# ALIGN USING indices
# ====================================================
aw_train = aw_tensor[idx_train]   # [N_train, 2]
aw_val   = aw_tensor[idx_val]     # [N_val, 2]
aw_test  = aw_tensor[idx_test]    # [N_test, 2]


# ====================================================
# CONCATENATE WITH SIGNAL EMBEDDINGS
# ====================================================
X_train_aug = torch.cat([X_train, aw_train], dim=1).float()
X_val_aug   = torch.cat([X_val,   aw_val],   dim=1).float()
X_test_aug  = torch.cat([X_test,  aw_test],  dim=1).float()

y_train = y_train.long()
y_val   = y_val.long()
y_test  = y_test.long()

# ====================================================
# 5. Few-Shot (pick K windows per class)
# ====================================================
K = 20  # shots per class

few_indices = []
for c in [0, 1, 2]:
    class_idx = (y_train == c).nonzero(as_tuple=True)[0]
    few_indices += class_idx[:min(K, len(class_idx))].tolist()

few_train_ds = TensorDataset(
    X_train_aug[few_indices], y_train[few_indices]
)

few_train_loader = DataLoader(few_train_ds, batch_size=32, shuffle=True)

val_ds = TensorDataset(X_val_aug, y_val)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# ====================================================
# 6. Linear Head Training (Few-Shot)
# ====================================================

class HydrationClassifier(nn.Module):
    def __init__(self, input_dim=2306, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 3),
        )

    def forward(self, x):
        return self.net(x)

model = HydrationClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

EPOCHS = 30
best_val_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    run_loss = 0

    for xb, yb in few_train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * yb.size(0)

    # ------------------------------------
    # Validation
    # ------------------------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, val_acc={val_acc:.4f}")

    # ------------------------------------
    # BEST MODEL SAVE
    # ------------------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict().copy()

# ====================================================
# SAVE BEST MODEL
# ====================================================
torch.save(best_state, "hydration_best_model.pth")
print("✔ Saved best model → hydration_best_model.pth")

torch.save({
    "model_state_dict": best_state,
    "input_dim": 2306,
    "num_classes": 3,
    "label_map": LABEL_MAP,
    "best_val_acc": best_val_acc
}, "hydration_best_model.pth")

print("✔️ Saved best model → hydration_best_model.pth")
print(f"Best validation accuracy: {best_val_acc:.4f}")

import torch
import torch.nn.functional as F

# =====================================================
# 1. Re-create model and load weights
# =====================================================
input_dim = X_test_aug.shape[1]   # should be 2306
num_classes = 3

ckpt = torch.load("hydration_best_model.pth", map_location="cpu")

model = HydrationClassifier(
    input_dim=ckpt["input_dim"],
    num_classes=ckpt["num_classes"]
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()



# =====================================================
# 2. Label mapping (inverse dict)
# =====================================================
inv_label_map = {
    0: "dehydrated_morning",
    1: "hydrated_afternoon",
    2: "mildly_hydrated_night"
}


# =====================================================
# 3. Inference loop
# =====================================================
correct = 0
total   = 0
predictions = []

with torch.no_grad():
    for i in range(X_test_aug.shape[0]):

        x = X_test_aug[i].unsqueeze(0)  # [1,2306]
        y = y_test[i].item()

        # forward pass
        logits = model(x)

        # softmax probabilities
        probs = F.softmax(logits, dim=1)   # [1,3]

        # predicted class
        pred_idx = probs.argmax(dim=1).item()
        conf     = probs.max().item()
        pred_lbl = inv_label_map[pred_idx]
        true_lbl = inv_label_map[y]

        # accuracy
        correct += int(pred_idx == y)
        total   += 1

        predictions.append({
            "pred_label": pred_lbl,
            "true_label": true_lbl,
            "confidence": conf
        })


# =====================================================
# 4. Final test accuracy
# =====================================================
test_acc = correct / total
print(f"\n=============================")
print(f"TEST ACCURACY: {test_acc:.4f}")
print("=============================\n")