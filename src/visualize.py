# show confusion matrix and label distribution for combined Sinhala and English datasets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

sns.set(style="whitegrid")

# Load datasets (small English subset)
from datasets import load_dataset

# Sinhala dataset
sinhala_ds = load_dataset("theekshana/sinhala-news-sentiment-classification")
sinhala_df = pd.DataFrame(sinhala_ds['train'])
sinhala_df = sinhala_df[['comment', 'simplified_label']].rename(columns={'comment':'text','simplified_label':'label'})

# English dataset (subset for speed)
english_ds = load_dataset("carant-ai/english_sentiment_dataset", split='train[:15000]')
english_df = pd.DataFrame(english_ds)
english_df = english_df[['text','label_text']].rename(columns={'label_text':'label'})
english_df['label'] = english_df['label'].str.capitalize()

# Combine
df = pd.concat([sinhala_df, english_df], ignore_index=True)

# Load model artifacts
svm_model = joblib.load("../models/svm.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
le = joblib.load("../models/label_encoder.pkl")

# Make predictions on entire combined dataset
X = vectorizer.transform(df['text'])
y_true = le.transform(df['label'])
y_pred = svm_model.predict(X)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Combined Dataset')
plt.show()

# Label distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title('Overall Label Distribution')
plt.show()

print("✅ Visualization complete")
