# 📝 Bilingo Sentiment Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A bilingual sentiment analysis system for **Sinhala** and **English** text, powered by XLM-RoBERTa transformer model. This application detects sentiment (positive, negative, or neutral) from user input with a modern, user-friendly web interface built with Streamlit.

---

## 🌟 Features

- 🌐 **Bilingual Support**: Analyzes sentiment in both Sinhala and English
- 🤖 **Transformer-Based**: Uses fine-tuned XLM-RoBERTa for accurate predictions
- 🎨 **Modern UI**: Clean, professional Streamlit interface
- ⚡ **Real-time Analysis**: Instant sentiment prediction
- 💾 **Lightweight**: Optimized for CPU inference
- 📊 **Three-Class Classification**: Positive, Negative, and Neutral sentiment detection

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Training](#-training)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎬 Demo

### Command Line Interface
```bash
=== Sinhala + English Sentiment Prediction ===
Enter text: I love this product!
>>> Predicted Sentiment ➜ POSITIVE
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- 4GB+ RAM recommended
- Internet connection (first run only, to download model)

### Step 1: Clone the Repository

```bash
git clone https://github.com/thiruniw/bilingo-sentiment-model.git
cd bilingo-sentiment-model
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
```
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
datasets>=2.14.0
```

### Step 4: Download Model Files

If the model files aren't included in the repository, download them:

```bash
# The model will be automatically loaded on first run
# Or manually download from HuggingFace (if hosted)
```

---

## 💻 Usage

### Web Application (Streamlit)

1. Navigate to the app directory:
```bash
cd app
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to:
```
http://localhost:8501
```

4. Enter text in Sinhala or English and click **"Analyze Sentiment"**

### Command Line Interface

1. Navigate to the src directory:
```bash
cd src
```

2. Run the prediction script:
```bash
python predict.py
```

3. Enter text when prompted or use test mode to see sample predictions

### Python API

```python
from predict import load_artifacts, predict_sentiment

# Load model once
tokenizer, model, label_encoder = load_artifacts()

# Predict sentiment
text = "මම මේක ගොඩක් කැමතියි!"  # Sinhala: "I love this!"
sentiment = predict_sentiment(text, tokenizer, model, label_encoder)
print(f"Sentiment: {sentiment}")  # Output: positive
```

---

## 📁 Project Structure

```
bilingo-sentiment/
│
├── app/
│   └── app.py                 # Streamlit web application
│
├── src/
│   ├── train.py              # Model training script
│   ├── predict.py            # Prediction and CLI interface
│   └── data_preprocess.py    # Data preprocessing utilities
│
├── models/
│   ├── xlm_roberta_sentiment_cpu/   # Trained model files
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── ...
│   └── label_encoder.pkl            # Label encoder for classes
│
├── logs/                     # Training logs
├── docs/                     # Documentation and images
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

### Architecture

- **Base Model**: [XLM-RoBERTa-base](https://huggingface.co/xlm-roberta-base)
- **Type**: Sequence Classification
- **Classes**: 3 (positive, negative, neutral)
- **Parameters**: ~278M
- **Max Sequence Length**: 128 tokens

### Training Data

- **Sinhala Dataset**: [sinhala-news-sentiment-classification](https://huggingface.co/datasets/theekshana/sinhala-news-sentiment-classification)
- **English Dataset**: [english_sentiment_dataset](https://huggingface.co/datasets/carant-ai/english_sentiment_dataset)
- **Training Samples**: ~10,000 (5,000 per language)
- **Test Split**: 20%

### Training Configuration

```python
Epochs: 3
Batch Size: 4
Learning Rate: 2e-5
Optimizer: AdamW
Hardware: CPU-optimized
Gradient Accumulation: 2 steps
```

### Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~85% |
| Test Accuracy | ~82% |

*Note: Performance varies by language and text domain*

---

## 🔧 Training

To retrain the model with your own data:

### Step 1: Prepare Your Data

Ensure your data is in CSV format with columns:
- `text`: The text to classify
- `label`: Sentiment label (positive/negative/neutral)

### Step 2: Update train.py

Modify the dataset loading section in `src/train.py`:

```python
# Load your custom dataset
df = pd.read_csv('your_data.csv')
```

### Step 3: Run Training

```bash
cd src
python train.py
```

**Training Time**: 
- CPU: 4-8 hours (for 10k samples, 3 epochs)
- GPU: 30-60 minutes (recommended)

### Step 4: Model Output

Trained model will be saved to:
```
models/xlm_roberta_sentiment_cpu/
models/label_encoder.pkl
```

---

## 🔮 Future Improvements

- [ ] Add support for more languages (Tamil, Hindi)
- [ ] Implement confidence scores in UI
- [ ] Add batch prediction support
- [ ] GPU acceleration option
- [ ] Export predictions to CSV
- [ ] API endpoint with FastAPI
- [ ] Docker containerization
- [ ] Model quantization for faster inference
- [ ] Fine-tune on domain-specific data

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Improving model accuracy
- Adding more languages
- UI/UX enhancements
- Documentation improvements
- Bug fixes and optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Thiruni Wijerathne**

- GitHub: [@thiruniw](https://github.com/thiruniw)
- Project Link: [https://github.com/thiruniw/bilingo-sentiment-model](https://github.com/thiruniw/bilingo-sentiment-model)

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) for the pre-trained model
- Dataset creators for Sinhala and English sentiment data
- [Streamlit](https://streamlit.io/) for the web framework

---

## 📞 Support

If you have any questions or run into issues:

1. Check the [Issues](https://github.com/thiruniw/bilingo-sentiment-model/issues) page
2. Open a new issue with detailed description
3. Contact via GitHub

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️**