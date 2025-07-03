# 🧠 Document Intelligence Transformer

A multimodal system built with PyTorch that classifies scanned documents by combining computer vision and OCR-based text understanding using a custom Transformer architecture.

---

## 🚀 Features

- 🖼 Image processing with ResNet18
- 🔤 Text processing from OCR using GloVe embeddings + TransformerEncoder
- 🔗 Fusion through gated TransformerEncoder
- 📊 Classification into 16 document types (Invoice, Resume, Form, Memo, etc.)
- 📈 Monitoring with Prometheus and Grafana
- 🌐 Web interface using Bootstrap + Chart.js
- 🐳 Full deployment with Docker Compose

---

## 🧱 Project Structure

document-intelligence-transformer/
├── api/ # Flask API for inference
├── webapp/ # Web interface with Bootstrap + Chart.js
├── training/ # Training scripts and synthetic dataset
├── model/ # Multimodal Transformer architecture
├── scripts/ # OCR preprocessing and vocabulary builder
├── monitoring/ # Prometheus + Grafana configs
├── docker-compose.yml # Full orchestration
├── requirements.txt
└── README.md


---

## 🛠️ Requirements

- Python 3.10+
- Tesseract OCR (`sudo apt install tesseract-ocr` or install manually on Windows/macOS)
- Docker + Docker Compose

---

## 📦 Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/ccaadaro/document-intelligence-transformer.git
cd document-intelligence-transformer
```

2. Launch all services

<pre> ```bash docker-compose up --build ``` </pre>
### 🔗 Access the Services

- **API**: [http://localhost:5000/predict](http://localhost:5000/predict)  
- **Web App**: [http://localhost:8000](http://localhost:8000)  
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000)  
  > Login: `admin` / `admin`

---

## 📥 API: `/predict`

### Input (`multipart/form-data`):
- `image`: scanned document (PNG/JPG)
- `text`: OCR-extracted plain text

### Example Output:
```json
{
  "document_type": "Invoice",
  "confidence": 0.9123,
  "probabilities": {
    "Invoice": 0.9123,
    "Resume": 0.0521,
    "Form": 0.0212,
    "...": "..."
  }
}
```
## 📊 Monitoring

- Prometheus scrapes real-time metrics from the Flask API (`/metrics`)
- Grafana visualizes:
  - ✅ Inference latency
  - ✅ Request frequency

---

## 💡 Use Cases

- Enterprise document classification
- Invoice/form/resume automation
- Real-world multimodal deep learning (Vision + Text fusion)

---

## 👤 Author

Developed as a portfolio-level machine learning deployment using PyTorch, Flask, Docker, and monitoring tools.

