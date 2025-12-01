# train_cpu_friendly.py

# Imports
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------------------
# 1️⃣ Load datasets
# ----------------------------

# Sinhala dataset
sinhala_ds = load_dataset("theekshana/sinhala-news-sentiment-classification")
sinhala_df = pd.DataFrame(sinhala_ds['train'])[['comment', 'simplified_label']].rename(columns={'comment':'text','simplified_label':'label'})

# English dataset (small subset)
english_ds = load_dataset("carant-ai/english_sentiment_dataset", split="train[:5000]")  # smaller for CPU
english_df = pd.DataFrame(english_ds)[['text','label_text']].rename(columns={'label_text':'label'})

# ✅ FIX: Use lowercase for consistency
english_df['label'] = english_df['label'].str.lower()

# Check label distribution before combining
print("\n" + "="*60)
print("LABEL DISTRIBUTION CHECK")
print("="*60)
print("\nSinhala dataset labels:")
print(sinhala_df['label'].value_counts())
print("\nEnglish dataset labels:")
print(english_df['label'].value_counts())

# Combine datasets (take only a subset to keep CPU usage low)
df = pd.concat([sinhala_df[:5000], english_df], ignore_index=True)  # limit Sinhala too

# ✅ Verify combined labels
print("\nCombined dataset labels:")
print(df['label'].value_counts())
print(f"\nUnique labels: {sorted(df['label'].unique())}")
print("="*60 + "\n")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ----------------------------
# 2️⃣ Encode labels
# ----------------------------
le = LabelEncoder()
train_labels = le.fit_transform(train_df['label'])
test_labels = le.transform(test_df['label'])

# ✅ Show label mapping
print("Label Encoder Mapping:")
for idx, label in enumerate(le.classes_):
    print(f"  {idx} → {label}")
print()

# Save label encoder
import os
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

# ----------------------------
# 3️⃣ Tokenizer
# ----------------------------
MODEL_NAME = "xlm-roberta-base"  # multilingual
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize
train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=128)

# ----------------------------
# 4️⃣ Torch Dataset
# ----------------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# ----------------------------
# 5️⃣ Model
# ----------------------------
num_labels = len(le.classes_)
print(f"Training model with {num_labels} labels: {list(le.classes_)}\n")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# ----------------------------
# 6️⃣ Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=3,                  # ✅ Increased from 1 to 3 for better learning
    per_device_train_batch_size=4,       # ✅ Increased from 2 to 4
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=200,
    save_strategy="epoch",               # ✅ Save each epoch
    evaluation_strategy="epoch",         # ✅ Evaluate each epoch
    report_to="none",
    no_cuda=True,
    fp16=False,
    dataloader_num_workers=0,
    load_best_model_at_end=True,         # ✅ Load best model
    metric_for_best_model="accuracy"
)

# ----------------------------
# 7️⃣ Metrics
# ----------------------------
from sklearn.metrics import accuracy_score, classification_report
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ----------------------------
# 8️⃣ Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ----------------------------
# 9️⃣ Train
# ----------------------------
print("Starting training...\n")
trainer.train()

# Evaluate
print("\nEvaluating on test set...")
results = trainer.evaluate()
print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")

# Get predictions for classification report
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
print("\nClassification Report:")
print(classification_report(test_labels, preds, target_names=le.classes_))

# Save final model
model.save_pretrained("models/xlm_roberta_sentiment_cpu")
tokenizer.save_pretrained("models/xlm_roberta_sentiment_cpu")

print("\n✅ CPU-friendly Transformer model trained and saved!")
print(f"Model saved to: models/xlm_roberta_sentiment_cpu")
print(f"Label encoder saved to: models/label_encoder.pkl")