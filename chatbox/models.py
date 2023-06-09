from django.db import models
from .config import (model_name_1, 
                     model_1, 
                     model_name_2, 
                     model_2,
                     tokenizer_1,
                     tokenizer_2)

# Generate text from model, tokenizer, and prompt
def _generate(model, tokenizer, input_text):
    tokens = tokenizer(input_text, return_tensors="pt")
    tokens_out = model.generate(**tokens, max_new_tokens=200)
    output_text = tokenizer.batch_decode(tokens_out, skip_special_tokens=True)
    return output_text

# Generate text from model, tokenizer, and prompt
def generate(model, input_prompt):
    generated_text = model(input_prompt)[0]['generated_text']
    return generated_text

# helper to find last occurrence of substring1 or substring2
def find_last_occurrence(text, substring1, substring2):
    index1 = text.rfind(substring1)
    index2 = text.rfind(substring2)
    
    if index1 == -1 and index2 == -1:
        return None
    elif index1 == -1:
        return substring2
    elif index2 == -1:
        return substring1
    else:
        return substring1 if index1 > index2 else substring2

# new output function
def new_output(output):
    last_message = output.split(': ')[-1]
    last_model = find_last_occurrence(output, model_name_1, model_name_2)
    new_model = model_name_2 if last_model == model_name_1 else model_name_1
    model = model_1 if new_model == model_name_1 else model_2
    input_text = last_message + '. Expand upon these thoughts and change the subject slightly.'
    if new_model == model_name_1:
        return new_model + ': ' + _generate(model, tokenizer_1, input_text)[0]
    elif new_model == model_name_2:
        return new_model + ': ' + _generate(model, tokenizer_2, input_text)[0]
    else:
        raise ValueError('Model not found.')