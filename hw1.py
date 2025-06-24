"""
This assignment involves building a question answering model using the Google T5-base model.
"""

import json
import nltk
import torch
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')

# context, answer, question, and ID format
# Read json dataset line by line
def load_qa_data(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_answer(question, context, tokenizer, model, device="cpu", max_len=64):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_bleu(predictions, references):
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    smooth = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        # 分词（适配中英文）
        ref_tokens = [nltk.word_tokenize(ref)]
        pred_tokens = nltk.word_tokenize(pred)

        bleu1.append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu2.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu3.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu4.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

    return {
        "BLEU-1": sum(bleu1) / len(bleu1),
        "BLEU-2": sum(bleu2) / len(bleu2),
        "BLEU-3": sum(bleu3) / len(bleu3),
        "BLEU-4": sum(bleu4) / len(bleu4),
    }


# Preprocessing and create the dataset
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_output_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"question: {item['question']} context: {item['context']}"
        target_text = item['answer']

        source = self.tokenizer(
            input_text, max_length=self.max_input_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        target = self.tokenizer(
            target_text, max_length=self.max_output_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(), # not fully understand
            'labels': target['input_ids'].squeeze()
        }


# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device) # fix the error here

train_data = load_qa_data("data/train.json")
test_data = load_qa_data("data/dev.json")

train_dataset = QADataset(train_data, tokenizer)
test_dataset = QADataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Train the model
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        # print(f"Loss: {total_loss / len(train_loader):.4f}")

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(train_loader):.4f}")


# model evaluation
model.eval()
test_data = load_qa_data("data/dev.json")  # 每个样本含 question, context, answer

predictions = []
references = []

for item in test_data:
    pred = generate_answer(item["question"], item["context"], tokenizer, model)
    predictions.append(pred)
    references.append(item["answer"])

bleu_scores = evaluate_bleu(predictions, references)

print("BLEU Scores:")
for k, v in bleu_scores.items():
    print(f"{k}: {v:.4f}")


