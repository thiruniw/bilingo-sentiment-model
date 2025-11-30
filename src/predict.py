import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

# ----------------------------
# Load model + tokenizer + label encoder
# ----------------------------
def load_artifacts():
    # Go up one level from src/ → project root
    root = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(root, "models", "xlm_roberta_sentiment_cpu")
    label_encoder_path = os.path.join(root, "models", "label_encoder.pkl")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    label_encoder = joblib.load(label_encoder_path)

    model.eval()  # set to evaluation mode
    return tokenizer, model, label_encoder


# ----------------------------
# Predict function
# ----------------------------
def predict_sentiment(text):
    tokenizer, model, label_encoder = load_artifacts()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Force CPU
    with torch.no_grad():
        outputs = model(**inputs)

    # Get highest probability class
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Decode label
    label = label_encoder.inverse_transform([predicted_class_id])[0]
    return label


# ----------------------------
# CLI Mode
# ----------------------------
if __name__ == "__main__":
    print("=== Sinhala + English Sentiment Prediction ===\n")
    while True:
        text = input("Enter text (or 'exit'): ")
        if text.lower() == "exit":
            break

        prediction = predict_sentiment(text)
        print(f"Predicted Sentiment ➜ {prediction}\n")
