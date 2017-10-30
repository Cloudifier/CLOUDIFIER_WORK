from django.utils.six import BytesIO
from rest_framework.parsers import JSONParser
from .serializers import CarDataSerializer

class CarsHelper:
    def get_cars(self, request):
        stream = BytesIO(request.body)
        data = JSONParser().parse(stream)
        response = {}
        response['responses'] = []
        it = 0
        for entry in data['data']:
            it = it + 1
            serializer = CarDataSerializer(data=entry)
            if serializer.is_valid():
                response['responses'].append({'status': 'CREATED', 'status_code': '200', 'description': 'SUCCESS'})
                print("Validated data for JSON obj #{0} is {1}"
                      .format(it, serializer.validated_data))
            else:
                print("Error encountered while processing JSON obj #{0}: {1}"
                      .format(it, serializer.errors))
                response['responses'].append({'status': 'BAD_REQUEST', 'status_code': '400', 'description': serializer.errors})

        return response

    def save_cars_to_db(self):
        pass