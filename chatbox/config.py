from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose model name from https://huggingface.co/models?filter=flax&pipeline_tag=translation
model_name_1 = "google/flan-t5-base"
model_name_2 = "MBZUAI/LaMini-Flan-T5-783M"

# Load tokenizer of model
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

# Load model from model_name
def load_model(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')

model_1 = load_model(model_name_1)
tokenizer_1 = load_tokenizer(model_name_1)
model_2 = load_model(model_name_2)
tokenizer_2 = load_tokenizer(model_name_2)