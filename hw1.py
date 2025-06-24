import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")

# print layerwise information
# print(json.dumps(list(model.keys())[:20], indent=4))
# training script

# test script
