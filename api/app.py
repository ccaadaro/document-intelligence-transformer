from flask import Flask, request, jsonify
import torch
from model.model import DocumentTransformer
from training.dataset import DocumentDataset  # para el vocabulario y procesamiento
import pickle
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DocumentTransformer()
model.load_state_dict(torch.load("api/model.pt", map_location=device))
model.eval().to(device)

# Tokenizador
with open("training/tokenizer.pkl", "rb") as f:
    vocab = pickle.load(f)

def tokenize(text, seq_len=32):
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens][:seq_len]
    if len(indices) < seq_len:
        indices += [vocab["<PAD>"]] * (seq_len - len(indices))
    return torch.tensor(indices)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_NAMES = ["Letter", "Form", "Email", "Handwritten", "Advertisement", "Scientific Report",
               "Scientific Publication", "Specification", "File Folder", "News Article",
               "Budget", "Invoice", "Presentation", "Questionnaire", "Resume", "Memo"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing image or text"}), 400

    image = Image.open(request.files["image"]).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    text = request.form["text"]
    text_tensor = tokenize(text).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor, text_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        response = {
            "document_type": CLASS_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
            }
        }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)