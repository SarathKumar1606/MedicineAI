import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU()
        )

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        features = self.cnn(x)

        # reshape → sequence
        b, c, h, w = features.size()

# collapse height dimension properly
        features = features.mean(2)        # [B, C, W]

# reshape to sequence
        features = features.permute(0, 2, 1)   # [B, W, C]

        # RNN
        output, _ = self.rnn(features)

        # logits
        output = self.fc(output)

        return output
    
class LabelConverter:
    def __init__(self):
        characters = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.char2idx = {c: i + 1 for i, c in enumerate(characters)}
        self.idx2char = {i + 1: c for i, c in enumerate(characters)}
        self.blank = 0

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, preds):
        result = ""
        prev = -1
        for p in preds:
            if p != prev and p != self.blank:
                result += self.idx2char.get(p, "")
            prev = p
        return result
    
   