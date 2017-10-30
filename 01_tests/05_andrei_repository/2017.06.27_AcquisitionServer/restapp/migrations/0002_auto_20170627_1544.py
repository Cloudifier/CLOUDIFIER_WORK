# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2017-06-27 15:44
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('restapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Entry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('car_id', models.IntegerField()),
                ('code', models.CharField(max_length=4)),
                ('value', models.CharField(max_length=50)),
                ('timestamp', models.DateField()),
            ],
        ),
        migrations.RemoveField(
            model_name='student',
            name='university',
        ),
        migrations.DeleteModel(
            name='Student',
        ),
        migrations.DeleteModel(
            name='University',
        ),
    ]
