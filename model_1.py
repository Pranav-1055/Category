import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_scheduler
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

class Args:
    input_csv = '/content/filtered_file.csv'
    checkpoint_dir = '/content/bert_checkpoints'
    epochs = 5

args = Args()

shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
os.makedirs(args.checkpoint_dir, exist_ok=True)

merged_path = os.path.join(args.checkpoint_dir, 'merged.csv')
if not os.path.exists(merged_path):
    chunksize = 1000
    for chunk in pd.read_csv(args.input_csv, chunksize=chunksize):
        label_cols = chunk.columns.difference(['text'])

        chunk[label_cols] = chunk[label_cols].fillna(0).astype(int)
        chunk[['text'] + label_cols.tolist()].to_csv(merged_path, mode='a', index=False, header=not os.path.exists(merged_path))

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=11):
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

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
sample_chunk = pd.read_csv(merged_path, nrows=1)
label_cols = sample_chunk.columns.difference(['text', 'html_title', 'h1', 'h2', 'p'])
model = BertMultiLabelClassifier(num_labels=len(label_cols))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()
epochs = args.epochs

chunk_resume_path = os.path.join(args.checkpoint_dir, 'last_chunk_checkpoint.pt')
resume_epoch, resume_chunk = 0, 0
if os.path.exists(chunk_resume_path):
    checkpoint = torch.load(chunk_resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch']
    resume_chunk = checkpoint['chunk_id'] + 1

chunk_size = 1000
num_chunks = sum(1 for _ in pd.read_csv(merged_path, chunksize=chunk_size))
best_f1 = 0.0

for epoch in range(resume_epoch, epochs):
    model.train()
    total_loss = 0
    chunk_iter = pd.read_csv(merged_path, chunksize=chunk_size)

    for chunk_id, chunk in enumerate(chunk_iter):
        if epoch == resume_epoch and chunk_id < resume_chunk:
            continue

        texts = chunk['text'].tolist()
        labels = chunk[label_cols].values
        dataset = CustomDataset(texts, labels, tokenizer, max_len=512)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader))

        loop = tqdm(loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1} Chunk {chunk_id+1}/{num_chunks}")
            loop.set_postfix(loss=loss.item())

        torch.save({
            'epoch': epoch,
            'chunk_id': chunk_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, chunk_resume_path)

    full_ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, full_ckpt_path)

    val_df = pd.read_csv(merged_path, skiprows=lambda i: i > 0 and i % 5 != 0)
    val_dataset = CustomDataset(val_df['text'].tolist(), val_df[label_cols].values, tokenizer, max_len=512)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Validation F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs('/content/bert_multilabel_model', exist_ok=True)

        torch.save(model.state_dict(), os.path.join('/content/bert_multilabel_model', 'pytorch_model.bin'))
        tokenizer.save_pretrained('/content/bert_multilabel_model')
