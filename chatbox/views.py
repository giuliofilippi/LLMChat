from django.shortcuts import render
from .models import new_output

# generate output
def generate_output(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        output = request.POST.get('output_text')
        if 'generate_button' in request.POST:
            updated_output = output + '\n' + '\n' + new_output(output)
        else:
            updated_output = 'Model 1: ' + input_text
    else:
        updated_output = ''

    return render(request, 'my_template.html', {'output': updated_output})
