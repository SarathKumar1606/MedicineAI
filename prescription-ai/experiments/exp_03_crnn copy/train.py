import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PrescriptionDataset
from src.ocr import CRNN, LabelConverter


# -------------------------
# Config (LIGHTWEIGHT)
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 12        # smaller = faster on CPU
EPOCHS = 5            # quick testing
DATASET_PATH = "dataset"


# -------------------------
# Load Dataset
# -------------------------
train_dataset = PrescriptionDataset(
    base_dir=DATASET_PATH,
    split="Training",
    augment=False       # OFF for speed (enable later)
)

# 🔥 Use small subset for fast testing
train_dataset.df = train_dataset.df.sample(1000, random_state=42)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -------------------------
# Model + Tools
# -------------------------
converter = LabelConverter()
num_classes = len(converter.char2idx) + 1  # + blank

model = CRNN(num_classes=num_classes).to(DEVICE)

criterion = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        images = batch["image"].to(DEVICE)
        texts = batch["label"]

        # Encode labels
        targets = []
        target_lengths = []

        for t in texts:
            encoded = converter.encode(t)
            targets.extend(encoded)
            target_lengths.append(len(encoded))

        targets = torch.tensor(targets, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        # Forward
        preds = model(images)            # [B, W, C]
        preds = preds.permute(1, 0, 2)   # [W, B, C]

        input_lengths = torch.full(
            size=(preds.size(1),),
            fill_value=preds.size(0),
            dtype=torch.long
        )

        # Loss
        loss = criterion(
            preds.log_softmax(2),
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ✅ update progress bar
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")


# -------------------------
# Save Model
# -------------------------
torch.save(model.state_dict(), "experiments/exp_03_crnn/model/crnn.pth")

print("\n✅ Training complete. Model saved.")