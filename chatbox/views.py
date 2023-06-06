from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer of model
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

# Load model from model_name
def load_model(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

model_name_1 = 'google/flan-t5-base'
model_name_2 = 'google/flan-t5-base'
model_1 = load_model(model_name_1)
tokenizer_1 = load_tokenizer(model_name_1)
model_2 = load_model(model_name_2)
tokenizer_2 = load_tokenizer(model_name_2)

# Create your views here.
def my_view(request):
    return render(request, 'my_template.html')

# Generate text from model, tokenizer, and prompt
def generate(model, tokenizer, input_text):
    tokens = tokenizer(input_text, return_tensors="pt")
    tokens_out = model.generate(**tokens, max_new_tokens=100)
    output_text = tokenizer.batch_decode(tokens_out, skip_special_tokens=True)
    return output_text

# new output function
def new_output(output):
    last_message = output.split(': ')[-1]
    last_model = output.split(':')[-2].split('\n')[-1]
    print (last_model)
    print (last_message)
    new_model = 'Model 2' if last_model == 'Model 1' else 'Model 1'
    model = model_1 if new_model == 'Model 1' else model_2
    tokenizer = tokenizer_1 if new_model == 'Model 1' else tokenizer_2
    input_text = last_message
    return  new_model + ': ' + generate(model, tokenizer, input_text)[0]

# generate output
def generate_output(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        output = request.POST.get('output_text')
        if 'generate_button' in request.POST:  # Check if the "Generate" button was clicked
            updated_output = output + '\n' + new_output(output)
        else:
            updated_output = 'Model 1: ' + input_text
    else:
        updated_output = ''

    return render(request, 'my_template.html', {'output': updated_output})
