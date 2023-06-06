from django.db import models
from .config import model_1, tokenizer_1, model_2, tokenizer_2

# Generate text from model, tokenizer, and prompt
def generate(model, tokenizer, input_text):
    tokens = tokenizer(input_text, return_tensors="pt")
    tokens_out = model.generate(**tokens, max_new_tokens=200)
    output_text = tokenizer.batch_decode(tokens_out, skip_special_tokens=True)
    return output_text

# new output function
def new_output(output):
    last_message = output.split(': ')[-1]
    last_model = output.split(':')[-2].split('\n')[-1]
    new_model = 'Model 2' if last_model == 'Model 1' else 'Model 1'
    model = model_1 if new_model == 'Model 1' else model_2
    tokenizer = tokenizer_1 if new_model == 'Model 1' else tokenizer_2
    input_text = last_message + '. Teach me something vaguely related to my previous sentence.'
    return  new_model + ': ' + generate(model, tokenizer, input_text)[0]