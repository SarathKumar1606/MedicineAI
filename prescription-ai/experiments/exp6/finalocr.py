import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "microsoft/trocr-base-handwritten"
JSON_PATH = "/content/medicines.json"
IMAGE_DIR = "/content/images/"

# -------------------------
# Load Processor + Model
# -------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)

# -------------------------
# Dataset Class
# -------------------------
class OCRDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(IMAGE_DIR + item["image"]).convert("RGB")
        text = item["text"]

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

dataset = OCRDataset(JSON_PATH)

# Split dataset
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

# -------------------------
# Training Arguments
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-medicine",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    fp16=torch.cuda.is_available()
)

# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    pred_texts = processor.batch_decode(preds, skip_special_tokens=True)
    label_texts = processor.batch_decode(labels, skip_special_tokens=True)

    correct = 0
    total = len(pred_texts)

    char_preds = []
    char_labels = []

    for p, l in zip(pred_texts, label_texts):
        if p.strip() == l.strip():
            correct += 1

        # Character-level comparison
        min_len = min(len(p), len(l))
        char_preds.extend(list(p[:min_len]))
        char_labels.extend(list(l[:min_len]))

    accuracy = correct / total

    return {
        "accuracy": accuracy
    }

# -------------------------
# Trainer
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# -------------------------
# Train
# -------------------------
trainer.train()

# -------------------------
# Evaluate
# -------------------------
predictions = trainer.predict(val_dataset)

pred_ids = predictions.predictions
label_ids = predictions.label_ids

pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

# -------------------------
# Confusion Matrix (Character-level)
# -------------------------
char_preds = []
char_labels = []

for p, l in zip(pred_texts, label_texts):
    min_len = min(len(p), len(l))
    char_preds.extend(list(p[:min_len]))
    char_labels.extend(list(l[:min_len]))


# -------------------------
# Save Model
# -------------------------
model.save_pretrained("./trocr-medicine-final")
processor.save_pretrained("./trocr-medicine-final")