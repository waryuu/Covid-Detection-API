import os
from google.cloud import storage

def get_bucket():
    client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    return client.bucket(bucket_name)

def upload_to_gcs(file_path, filename):
    bucket = get_bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path)
    blob.make_public()
    return blob.public_url

def validate_metadata(age, gender, height, weight):
    return all([
        isinstance(age, int) and age > 0,
        gender in ("male", "female"),
        isinstance(height, float) and height > 0,
        isinstance(weight, float) and weight > 0
    ])

def delete_from_gcs(filename):
    bucket = get_bucket()
    blob = bucket.blob(filename)
    if blob.exists():
        blob.delete()
        return True
    return False
