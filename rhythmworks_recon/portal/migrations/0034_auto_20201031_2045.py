# Generated by Django 3.1.2 on 2020-10-31 15:15

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0033_tbl_workflow_activity_mst'),
    ]

    operations = [
        migrations.CreateModel(
            name='tbl_assign_pages_roles',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('share_id', models.IntegerField(default=0, verbose_name='Share ID')),
                ('assigned_to_role', models.IntegerField(default=0, verbose_name='Assigned To Role')),
                ('parent_code', models.CharField(max_length=30, verbose_name='Parent Code')),
                ('child_code', models.CharField(max_length=30, verbose_name='Child Code')),
                ('sub_child_code', models.CharField(max_length=30, verbose_name='Sub Child Code')),
                ('read_access', models.CharField(default='N', max_length=1, verbose_name='Read Access')),
                ('write_edit_access', models.CharField(default='N', max_length=1, verbose_name='Write Edit Access')),
                ('delete_access', models.CharField(default='N', max_length=1, verbose_name='Read Access')),
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
        migrations.CreateModel(
            name='tbl_assign_role_user',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('share_id', models.IntegerField(default=0, verbose_name='Share ID')),
                ('assigned_to_user', models.CharField(max_length=30, verbose_name='Assigned To User')),
                ('is_active', models.CharField(default='Y', max_length=1, verbose_name='Is Active')),
                ('is_deleted', models.CharField(default='N', max_length=1, verbose_name='Is Deleted')),
                ('created_date_time', models.DateTimeField(default=django.utils.timezone.localtime, verbose_name='Created Date Time')),
                ('created_by', models.IntegerField(default=0, verbose_name='Created By')),
                ('updated_date_time', models.DateTimeField(default=django.utils.timezone.localtime, verbose_name='Updated Date Time')),
                ('updated_by', models.IntegerField(default=0, verbose_name='Updated By')),
                ('sub_application_id', models.CharField(max_length=20, verbose_name='Sub-Application ID')),
                ('application_id', models.CharField(max_length=20, verbose_name='Application ID')),
                ('assigned_to_role_ref_id', models.ForeignKey(default=0, on_delete=django.db.models.deletion.PROTECT, to='portal.tbl_role_mst', verbose_name='Assigned to Role Ref Id')),
            ],
        ),
        migrations.CreateModel(
            name='tbl_left_panel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('share_id', models.IntegerField(default=0, verbose_name='Share ID')),
                ('form_name', models.CharField(max_length=30, verbose_name='Form Name')),
                ('form_link', models.CharField(default=0, max_length=200, verbose_name='Form Link')),
                ('is_parent', models.CharField(default='N', max_length=1, verbose_name='Is Parent')),
                ('is_child', models.CharField(default='N', max_length=1, verbose_name='Is Child')),
                ('is_sub_child', models.CharField(default='N', max_length=1, verbose_name='Is Sub Child')),
                ('parent_code', models.CharField(max_length=30, verbose_name='Parent Code')),
                ('child_code', models.CharField(max_length=30, verbose_name='Child Code')),
                ('sub_child_code', models.CharField(max_length=30, verbose_name='Sub Child Code')),
                ('icon_class', models.CharField(max_length=30, verbose_name='Icon Class')),
                ('sequence_id', models.IntegerField(default=0, verbose_name='Sequence ID')),
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
        migrations.AddField(
            model_name='tbl_assign_pages_roles',
            name='form_ref_id',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.PROTECT, to='portal.tbl_left_panel', verbose_name='Form Ref Id'),
        ),
    ]