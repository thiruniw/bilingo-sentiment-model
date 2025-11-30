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
english_df['label'] = english_df['label'].str.capitalize()

# Combine datasets (take only a subset to keep CPU usage low)
df = pd.concat([sinhala_df[:5000], english_df], ignore_index=True)  # limit Sinhala too
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ----------------------------
# 2️⃣ Encode labels
# ----------------------------
le = LabelEncoder()
train_labels = le.fit_transform(train_df['label'])
test_labels = le.transform(test_df['label'])

# Save label encoder
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
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# ----------------------------
# 6️⃣ Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=1,                 # only 1 epoch for CPU
    per_device_train_batch_size=8,      # smaller batch
    per_device_eval_batch_size=8,
    logging_dir="logs",
    logging_steps=50,
    save_strategy="no",                 # skip intermediate saves
    report_to="none",                   # disable wandb
    learning_rate=2e-5,
    no_cuda=True                         # force CPU
)

# ----------------------------
# 7️⃣ Metrics
# ----------------------------
from sklearn.metrics import accuracy_score
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
trainer.train()

# Save final model
model.save_pretrained("models/xlm_roberta_sentiment_cpu")
tokenizer.save_pretrained("models/xlm_roberta_sentiment_cpu")

print("✅ CPU-friendly Transformer model trained and saved!")
