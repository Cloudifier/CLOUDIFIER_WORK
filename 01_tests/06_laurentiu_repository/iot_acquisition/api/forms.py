from django import forms

class SearchForm(forms.Form):
  CarID = forms.IntegerField(required=False)
  Code = forms.CharField(max_length=100, required=False)