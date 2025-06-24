"""
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
"""

import json
import torch
import torch.nn as nn

model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))