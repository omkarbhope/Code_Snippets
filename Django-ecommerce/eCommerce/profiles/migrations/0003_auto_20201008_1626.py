# Generated by Django 3.1.2 on 2020-10-08 10:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0002_profile_descrition'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='descrition',
            field=models.TextField(default='descrition default text'),
        ),
    ]
