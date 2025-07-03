import os
import pytesseract
from PIL import Image

def ocr_image(image_path):
    image = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, fname)
            text = ocr_image(image_path)
            out_name = os.path.splitext(fname)[0] + ".txt"
            with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ OCR extracted: {fname}")

if __name__ == "__main__":
    input_dir = "data/raw_documents/"
    output_dir = "data/train_texts/"
    process_folder(input_dir, output_dir)