# Generated by Django 3.1.2 on 2020-10-29 11:36

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0030_auto_20201029_1458'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tbl_reconcilation_process_details',
            name='header_ref_id',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.PROTECT, related_name='initialItemRow', to='portal.tbl_reconcilation_process_mst', verbose_name='Header Ref ID'),
        ),
    ]