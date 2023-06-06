from django.shortcuts import render

# Create your views here.
def my_view(request):
    return render(request, 'my_template.html')

# new output function
def new_output(output):
    last_message = output.split(': ')[-1]
    last_model = output.split(':')[0]
    new_model = 'Model 2' if last_model == 'Model 1' else 'Model 1'
    return new_model + ': ' + 'haha'

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
