# Generated by Django 3.1.2 on 2020-11-03 13:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0037_tbl_sourcetable_fields_mst'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tbl_sourcetable_fields_mst',
            name='section_identifier_id',
            field=models.CharField(max_length=20, verbose_name='Section Identifier ID'),
        ),
    ]
