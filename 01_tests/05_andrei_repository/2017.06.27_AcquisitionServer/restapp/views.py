from .models import Entry

from restapp.serializers import EntrySerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django.views import View
from django.views.generic import TemplateView

import pandas as pd

class HomeView(TemplateView):
    template_name = "about.html"


# clasa iostorage
i = 0
df = pd.DataFrame(columns = ["CarID", "Code", "Value", "TimeStamp"])

class EntriesList(APIView):
    """
    List all entries, or post a new entry.
    """ 
    def get(self, request, format=None):
        entries = Entry.objects.all()
        serializer = EntrySerializer(entries, many=True)

        return Response(serializer.data)

    def post(self, request, format=None):
        global df
        global i

        serializer = EntrySerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()

            items = list(serializer.validated_data.items())
            values = [snd for _,snd in items]
            df.loc[i] = values
            i += 1
            print(df.head())

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, format=None):
        global df
        global i

        entries = Entry.objects.all()
        entries.delete()

        i = 0
        df.iloc[i:i]

        return Response(status=status.HTTP_204_NO_CONTENT)