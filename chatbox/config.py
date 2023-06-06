from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          T5Tokenizer,
                          T5ForConditionalGeneration)

# Choose model name from https://huggingface.co/models?filter=flax&pipeline_tag=translation
model_name_1 = "MBZUAI/LaMini-Flan-T5-783M"
model_name_2 = "google/flan-t5-base"

# Load tokenizer of model
def load_tokenizer(model_name):
    if model_name == "MBZUAI/LaMini-Flan-T5-783M":
        return AutoTokenizer.from_pretrained(model_name)
    elif model_name == "google/flan-t5-base":
        return T5Tokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Tokenizer not found for model: " + model_name)

# Load model from model_name
def load_model(model_name):
    if model_name == "MBZUAI/LaMini-Flan-T5-783M":
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
    elif model_name == "google/flan-t5-base":
        return T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError("Model not found for model: " + model_name)

model_1 = load_model(model_name_1)
tokenizer_1 = load_tokenizer(model_name_1)
model_2 = load_model(model_name_2)
tokenizer_2 = load_tokenizer(model_name_2)