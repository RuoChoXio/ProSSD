import torch
from transformers import AutoModel, AutoTokenizer

def build_model(model_name_or_path: str, device: str = "cpu"):
    print(f"[Model] Loading: {model_name_or_path} to {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    return tokenizer, model
