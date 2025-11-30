import joblib
import os

def load_artifacts():
    """Load the ML model and preprocessing artifacts"""
    base_path = os.path.join(os.path.dirname(__file__), "../models")
    model = joblib.load(os.path.join(base_path, "svm.pkl"))
    vectorizer = joblib.load(os.path.join(base_path, "vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(base_path, "label_encoder.pkl"))
    return model, vectorizer, label_encoder

def predict_sentiment(text):
    """Predict sentiment"""
    model, vectorizer, label_encoder = load_artifacts()

    X = vectorizer.transform([text])
    pred = model.predict(X)
    label = label_encoder.inverse_transform(pred)[0].capitalize()  # First letter capital
    return label

if __name__ == "__main__":
    print("=== Sinhala & English Sentiment Prediction ===\n")
    while True:
        user_text = input("Enter text (or 'exit'): ")
        if user_text.lower() == "exit":
            break
        prediction = predict_sentiment(user_text)
        print(f"Predicted Sentiment ➜ {prediction}\n")
