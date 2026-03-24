import pandas as pd
import os
import json

BASE_DIR = "prescription-ai/dataset"

meds = set()

for split in ["Training", "Validation", "Testing"]:
    path = os.path.join(BASE_DIR, split)
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    
    df = pd.read_csv(os.path.join(path, csv_file))
    meds.update(df["MEDICINE_NAME"].str.lower().str.strip())

vocab = sorted(list(meds))

with open("medicine_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print("Saved vocab:", len(vocab))