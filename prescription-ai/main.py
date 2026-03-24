from src.infer import CRNNInference

MODEL_PATH = "experiments/exp_03_crnn/model/crnn.pth"

# load model
ocr = CRNNInference(MODEL_PATH)

# test image (change index if needed)
test_image = "dataset/Testing/testing_words/m.png"

prediction = ocr.predict_from_path(test_image)

print("\nPrediction:", prediction)