from django import  forms

class contactForm(forms.Form):
    name = forms.CharField(max_length=100, required=False, help_text='100 characters max')
    email = forms.EmailField(required = True)
    comment = forms.CharField(required = True, widget=forms.Textarea)