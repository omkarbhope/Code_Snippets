# Generated by Django 3.1.2 on 2020-10-08 11:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0004_profile_location'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='job',
            field=models.CharField(max_length=120, null=True),
        ),
    ]
