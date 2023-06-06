from transformers import (AutoTokenizer, 
                          AutoModel,
                          AutoModelForSeq2SeqLM,
                          BartConfig,
                          pipeline)
import torch

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose model name from https://huggingface.co/models
model_name_1 = "MBZUAI/LaMini-Flan-T5-783M"
model_name_2 =  "MBZUAI/LaMini-GPT-1.5B"

# Load tokenizer of model
def load_tokenizer(model_name):
    if model_name == "MBZUAI/LaMini-Flan-T5-783M":
        return AutoTokenizer.from_pretrained(model_name)
    if model_name == "MBZUAI/LaMini-GPT-1.5B":
        return AutoTokenizer.from_pretrained(model_name)
    elif model_name == "google/flan-t5-base":
        return AutoTokenizer.from_pretrained(model_name)
    elif model_name == "google/flan-t5-large":
        return AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Tokenizer not found for model: " + model_name)

# Load model from model_name
def _load_model(model_name):
    if model_name == "MBZUAI/LaMini-Flan-T5-783M":
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if model_name == "MBZUAI/LaMini-GPT-1.5B":
        return BartConfig.from_pretrained(model_name)
    elif model_name == "google/flan-t5-base":
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_name == "google/flan-t5-large":
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise ValueError("Model not found for model: " + model_name)
    
# new load model
def load_model(model_name):
    if model_name == "MBZUAI/LaMini-GPT-1.5B":
        return pipeline('text-generation', model = model_name)
    elif model_name == "MBZUAI/LaMini-Flan-T5-783M":
        return pipeline('text2text-generation', model = model_name)
    else:
        raise ValueError("Model not found for model: " + model_name)

# models and tokenizers
model_1 = load_model(model_name_1)
tokenizer_1 = load_tokenizer(model_name_1)
model_2 = load_model(model_name_2)
tokenizer_2 = load_tokenizer(model_name_2)