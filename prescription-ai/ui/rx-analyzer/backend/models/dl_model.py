# dl_model.py
# -------------------------------------------------------
# Drop your trained DL model inference code here.
# The main.py calls run_inference(image_path: str) -> dict
# -------------------------------------------------------

def run_inference(image_path: str) -> dict:
    """
    Load your model and run inference on the given image path.
    Return a dict matching the prescription JSON schema used in main.py.

    Example skeleton:
        import torch
        from torchvision import transforms
        from PIL import Image

        model = load_your_model("weights.pt")
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image)
        output = model(tensor)
        return parse_output(output)
    """
    raise NotImplementedError("Replace this with your actual model inference logic.")
