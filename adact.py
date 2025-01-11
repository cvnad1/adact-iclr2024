import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

class Ace(nn.Module):
    def __init__(self) -> None:
        super(Ace, self).__init__()
        self.w = nn.Parameter(torch.randn(5))
        self.activations = [F.relu, torch.sigmoid, torch.tanh, F.silu, F.gelu]
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        fan_in = len(self.activations)
        fan_out = 1
        std = torch.sqrt(torch.tensor(2.0) / (fan_in + fan_out))
        nn.init.normal_(self.w, mean=0.0, std=std.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_weights = F.softmax(self.w, dim=0)
        weighted_activations = torch.stack([w * activation(x) for w, activation in zip(normalized_weights, self.activations)], dim=0)
        return weighted_activations.sum(dim=0)

class ActivationWrapper(nn.Module):
    def __init__(self, activation_fn: nn.Module) -> None:
        super(ActivationWrapper, self).__init__()
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(x)

def load_model(model_name: str, num_labels: int, adact: bool = False) -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    if adact:
        for i in range(12):
            model.bert.encoder.layer[i].intermediate.intermediate_act_fn = Ace()
    return model, tokenizer

def adact_opt(model: nn.Module) -> tuple:
    activations = [F.relu, torch.sigmoid, torch.tanh, F.silu, F.gelu]
    for i in range(12):
        ace_instance = model.bert.encoder.layer[i].intermediate.intermediate_act_fn
        max_index = torch.argmax(ace_instance.w).item()
        selected_activation = activations[max_index]
        model.bert.encoder.layer[i].intermediate.intermediate_act_fn = ActivationWrapper(selected_activation)
    chosen = [model.bert.encoder.layer[i].intermediate.intermediate_act_fn.activation_fn.__name__ for i in range(12)]
    return model, chosen
