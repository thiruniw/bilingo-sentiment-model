import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from pathlib import Path
import warnings
import os
import sys

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------
# Load model + tokenizer + label encoder
# ----------------------------
def load_artifacts():
    root = Path(__file__).parent.parent.absolute()
    
    model_path = root / "models" / "xlm_roberta_sentiment_cpu"
    label_encoder_path = root / "models" / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_safetensors=True
    )
    
    label_encoder = joblib.load(label_encoder_path)
    model.eval()
    
    return tokenizer, model, label_encoder


def predict_sentiment(text, tokenizer, model, label_encoder, show_details=False):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_class_id = torch.argmax(logits, dim=1).item()
    label = label_encoder.inverse_transform([predicted_class_id])[0]
    
    if show_details:
        print(f"\n--- Prediction Details ---")
        print(f"Text: '{text}'")
        print(f"Logits: {[f'{x:.4f}' for x in logits[0].tolist()]}")
        print(f"Probabilities:")
        for idx, prob in enumerate(probabilities):
            class_name = label_encoder.inverse_transform([idx])[0]
            print(f"  {class_name}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
        print(f"Predicted: {label}")
    
    return label


# ----------------------------
# CLI Mode with Diagnostics
# ----------------------------
def main():
    print("=" * 60)
    print("=== Sinhala + English Sentiment Prediction ===")
    print("=" * 60)
    print()
    
    # Pre-load model once
    print("Loading model...", end=" ", flush=True)
    tokenizer, model, label_encoder = load_artifacts()
    print("✓ Done\n")
    
    # Show model info
    print("-" * 60)
    print("MODEL INFORMATION")
    print("-" * 60)
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Label classes: {list(label_encoder.classes_)}")
    print(f"Label mapping: {dict(enumerate(label_encoder.classes_))}")
    print()
    
    # Test with neutral examples
    print("-" * 60)
    print("TESTING WITH SAMPLE TEXTS")
    print("-" * 60)
    
    test_texts = [
        "I love this!",  # Positive
        "I hate this!",  # Negative
        "This is a book.",  # Neutral
        "The sky is blue.",  # Neutral
        "It is what it is.",  # Neutral
        "එය පොතක්",  # Neutral (Sinhala: It is a book)
    ]
    
    for text in test_texts:
        predict_sentiment(text, tokenizer, model, label_encoder, show_details=True)
        print()
    
    # Interactive mode
    print("=" * 60)
    print("INTERACTIVE MODE (type 'exit' to quit)")
    print("=" * 60)
    print()
    
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() == "exit":
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue

            prediction = predict_sentiment(text, tokenizer, model, label_encoder, show_details=True)
            print(f"\n>>> Predicted Sentiment ➜ {prediction}\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()