# Generated by Django 3.1.2 on 2020-10-29 09:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0029_auto_20201029_1457'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tbl_source_details',
            name='header_ref_id',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, related_name='initialItemRow', to='portal.tbl_source_mst', verbose_name='Header Ref ID'),
        ),
    ]