import json
import nltk
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_qa_data(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_answer(question, context, tokenizer, model, device="cuda", max_len=64):
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
        pred_tokens = list(pred.strip())
        ref_tokens = [list(r.strip()) for r in references]
        # ref_tokens = [nltk.word_tokenize(ref)]
        # pred_tokens = nltk.word_tokenize(pred)

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


'''
model evaluation

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("saved_model_t5_base")
tokenizer = T5Tokenizer.from_pretrained("saved_model_t5_base")
model = model.to(device)
model.eval()
test_data = load_qa_data("data/dev.json")  # 每个样本含 question, context, answer

predictions = []
references = []

for i,item in enumerate(test_data):
    pred = generate_answer(item["question"], item["context"], tokenizer, model)
    predictions.append(pred)
    references.append(item["answer"])

bleu_scores = evaluate_bleu(predictions, references)

print("BLEU Scores:")
for k, v in bleu_scores.items():
    print(f"{k}: {v:.4f}")

'''
Given context and question, provide the answer
'''
answer = generate_answer(
    "What is the capital of France?",
    "France is a country in Europe. Its capital city is Paris.",
    tokenizer, model, device
)
print("Answer:", answer)

answer = generate_answer(
    "2017年银行贷款基准利率",
    "年基准利率4.35%。 从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。",
    tokenizer, model, device
)
print("Answer:", answer)

# BLEU Scores:
# BLEU-1: 0.5655
# BLEU-2: 0.5372
# BLEU-3: 0.4242
# BLEU-4: 0.3478