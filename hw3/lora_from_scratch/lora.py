
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import AutoModel, AutoModelForCausalLM


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,      # full model linear layer
        rank: int = 8,              # lora rank
        alpha: int = 16,            # lora alpha
        dropout_p: float = 0.0,     # lora dropout
        test_mode: bool = False,
    ):   
        super(LoraLinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer) # 关键点
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)
        
        # 创建lora矩阵 [nn.linear()]
        # In pytorch, nn.Linear(in_features, out_features) represents
        #                    y = xW^T + b,
        # where W is of shape (out_features, in_features).
        self.lora_A = nn.Parameter(torch.empty((rank, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, rank), dtype=base_layer.weight.dtype))
        
        # 初始化lora矩阵
        torch.nn.init.normal_(self.lora_A, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lora_B, mean=0.0, std=1.0)
        
        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        scale = float(self.alpha)/float(self.rank)
        lora_adjustment = F.linear(x, self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        
        return self.base_layer(x) + lora_adjustment * scale


# 找到Lora中所有层并逐层替换
def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout_p: float = 0.0,
    embed_requires_grad: bool = False,      # embedding 层是否训练
    norm_requires_grad: bool = False,       # norm 层是否训练
    head_requires_grad: bool = False,       # lm_head 层是否训练（Causal LM才有）???
    test_mode: bool = False,                # 测试模式，用于控制 lora_B 是否为全零
):
    # what else? what else?
    
    for name, child in module.named_children():
        # 先处理额外的层，lm_head 也是 linear，所以先处理
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad
        # 替换所有线性层，QLoRA 做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, rank=rank, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            setattr(module, name, lora_linear)
        # 递归向下替换
        else:
            replace_linear_with_lora(
                child, rank, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                test_mode=test_mode
            )


def print_trainable_parameters(model: nn.Module):
    """
    打印可训练参数，表现和 PeftModel 的 print_trainable_parameters 方法类似
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params
    
    # 返回可训练参数量、所有参数量、可训练参数量占比（百分比）
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")


config = AutoConfig.for_model('llama')
config.hidden_size = 24
config.intermediate_size = config.hidden_size * 4
config.num_attention_heads = 4
config.num_hidden_layers = 4
config.num_key_value_heads = 2
config.vocab_size = 128

# raw_model = AutoModel.from_config(config)  # 没带因果头
raw_model = AutoModelForCausalLM.from_config(config)  # 带了因果头
# print(raw_model)
# print_trainable_parameters(raw_model)

lora_model = copy.deepcopy(raw_model)  # 深克隆，独立一个新模型
replace_linear_with_lora(lora_model, rank=8, alpha=16)  # 替换
print_trainable_parameters(lora_model) # 打印参数情况
print(lora_model)
    
