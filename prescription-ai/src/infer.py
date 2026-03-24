import torch
from PIL import Image
from torchvision import transforms

from src.ocr import CRNN, LabelConverter


class CRNNInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Label converter
        self.converter = LabelConverter()
        num_classes = len(self.converter.char2idx) + 1

        # Load model
        self.model = CRNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor()
        ])

        print("[INFO] CRNN model loaded.")

    def predict(self, image):
        # Convert to grayscale
        image = image.convert("L")

        # Transform
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Forward
        with torch.no_grad():
            preds = self.model(image)

        preds = preds.permute(1, 0, 2)  # [W, B, C]
        preds = torch.argmax(preds, dim=2)

        preds = preds[:, 0].cpu().numpy()

        # Decode
        text = self.converter.decode(preds)

        return text

    def predict_from_path(self, image_path):
        image = Image.open(image_path)
        return self.predict(image)