import argparse
import os
from src.ocr import hybrid_ocr
from src.matcher import match_medicines


def run_pipeline(image_path, output_dir="outputs/predictions"):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📄 Processing: {image_path}")

    # ---- Step 1: OCR ----
    print("\n🔍 Running OCR...")
    extracted_text = hybrid_ocr(image_path)

    print("\n----- OCR TEXT -----\n")
    print(extracted_text)

    # ---- Step 2: Medicine Matching ----
    print("\n💊 Matching medicines...")
    medicines = match_medicines(extracted_text)

    print("\n----- MATCHED MEDICINES -----\n")
    for med in medicines:
        print(f"- {med}")

    # ---- Step 3: Save Output ----
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, filename + ".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("OCR TEXT:\n")
        f.write(extracted_text + "\n\n")

        f.write("MATCHED MEDICINES:\n")
        for med in medicines:
            f.write(f"- {med}\n")

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prescription AI Pipeline")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to prescription image")

    args = parser.parse_args()

    run_pipeline(args.image)