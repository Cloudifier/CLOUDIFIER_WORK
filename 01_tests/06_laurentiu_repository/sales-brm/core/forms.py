from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
	username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'User'}),required = True)
	first_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Prenume'}),required = False)
	last_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Nume Familie'}),required = False)
	industry = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Industrie'}),required = False)
	email = forms.EmailField(widget=forms.TextInput(attrs={'placeholder': 'Email'}),required = True)
	password1 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Parola'}),required = True)
	password2 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Confirmare Parola'}),required = True)

	class Meta:
		model = User
		fields = ('username', 'first_name', 'last_name', 'industry', 'email', 'password1', 'password2', )


"""
username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Nume utilizator *'}),required = True)
first_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Nume familie'}),required = False)
last_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Prenume'}),required = False)
industry = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Industrie'}),required = False)
email = forms.EmailField(widget=forms.TextInput(attrs={'placeholder': 'Email *'}),required = True)
password1 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Parola *'}),required = True)
password2 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Confirmare Parola *'}),required = True)
"""