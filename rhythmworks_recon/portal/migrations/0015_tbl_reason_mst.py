# Generated by Django 3.1.2 on 2020-10-22 17:31

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0014_tbl_api_definition_mst_self_channel_ref_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='tbl_reason_mst',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False, verbose_name='ID')),
                ('share_id', models.IntegerField(default=0, verbose_name='Share ID')),
                ('rej_reason_code', models.CharField(max_length=20, verbose_name='Rejection Reason Code')),
                ('rej_reason', models.TextField(verbose_name='Rejection Reason')),
                ('is_active', models.CharField(default='Y', max_length=1, verbose_name='Is Active')),
                ('is_deleted', models.CharField(default='N', max_length=1, verbose_name='Is Deleted')),
                ('created_date_time', models.DateTimeField(default=django.utils.timezone.localtime, verbose_name='Created Date Time')),
                ('created_by', models.IntegerField(default=0, verbose_name='Created By')),
                ('updated_date_time', models.DateTimeField(default=django.utils.timezone.localtime, verbose_name='Updated Date Time')),
                ('updated_by', models.IntegerField(default=0, verbose_name='Updated By')),
                ('sub_application_id', models.CharField(max_length=20, verbose_name='Sub-Application ID')),
                ('application_id', models.CharField(max_length=20, verbose_name='Application ID')),
            ],
        ),
    ]