from django.db import migrations
import security.fields

class Migration(migrations.Migration):
    dependencies = [("anomaly_detection", "0001_initial")]
    operations = [
        migrations.AlterField(model_name="userbehaviour", name="user_agent", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="userbehaviour", name="location", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="userbehaviour", name="device_id", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="userbehaviour", name="activity_data", field=security.fields.EncryptedJSONField(blank=True, default=dict)),
        migrations.AlterField(model_name="loginactivity", name="user_agent", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="loginactivity", name="location", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="loginactivity", name="device_id", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="loginactivity", name="failure_reason", field=security.fields.EncryptedTextField(blank=True)),
        migrations.AlterField(model_name="anomaly", name="explanation", field=security.fields.EncryptedTextField(blank=True)),
    ]
