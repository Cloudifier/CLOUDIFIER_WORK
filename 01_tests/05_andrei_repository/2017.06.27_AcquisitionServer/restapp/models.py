from django.db import models

class Entry(models.Model):
    car_id = models.IntegerField()
    code = models.CharField(max_length=4)
    value = models.CharField(max_length=50)
    timestamp = models.DateField()

    def __str__(self):
        return str(self.car_id) ++ " " ++ self.code