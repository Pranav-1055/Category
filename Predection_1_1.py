import os
import torch
import requests
import re
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from torch import nn
from torch.nn.functional import sigmoid

nltk.download('stopwords')

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load label columns from training CSV
merged_csv_path = '/content/bert_checkpoints/merged.csv'
label_cols = pd.read_csv(merged_csv_path, nrows=1).columns.difference(['text', 'html_title', 'h1', 'h2', 'p']).tolist()

# ✅ Define model architecture (same as training)
from transformers import BertModel

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=10):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# ✅ Load tokenizer and model from saved folder
model_dir = '/content/bert_multilabel_model'
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = BertMultiLabelClassifier(num_labels=len(label_cols))
model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=device))
model.to(device)
model.eval()

# ✅ Text cleaning utilities
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return " ".join(word for word in words if word not in stop_words)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return remove_stopwords(text)

# ✅ Web scraping + cleaning
def scrape_and_clean(url, custom_tags=['h1', 'h2', 'p']):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url} => {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    texts = []
    if soup.title:
        texts.append(soup.title.get_text(strip=True))

    for tag in custom_tags:
        elements = soup.find_all(tag)
        texts.extend([elem.get_text(strip=True) for elem in elements])

    full_text = " ".join(texts)
    cleaned = clean_text(full_text)
    tokenized = tokenizer(cleaned, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

    return {
        "clean_text": cleaned,
        "tokenized": tokenized
    }

# ✅ Prediction function
def predict_website_category(url):
    result = scrape_and_clean(url)
    if not result:
        return

    input_ids = result['tokenized']['input_ids'].to(device)
    attention_mask = result['tokenized']['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = sigmoid(logits).cpu().numpy()[0]

    print(f"\n🔗 URL: {url}")
    print(f"\n🧼 Cleaned Text (first 500 chars):\n{result['clean_text'][:500]}...\n")
    print("📊 Predicted Labels and Scores:")
    for label, score in zip(label_cols, probs):
        print(f"{label:<40}: {score:.4f}")

# ✅ Run prediction on example URL
if __name__ == "__main__":
    test_url = "https://app.intigriti.com/researcher/programs/vrtnv/vrtnv/detail"
    predict_website_category(test_url)
