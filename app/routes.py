from flask import Blueprint, request, jsonify
import uuid
import os

from app.utils import (
    validate_metadata,
    upload_to_gcs,
    delete_from_gcs
)
from app.ml_service import run_inference

api = Blueprint("api", __name__)

@api.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    test_id = request.form.get("test_id")
    age = request.form.get("age", type=int)
    gender = request.form.get("gender")
    height = request.form.get("height", type=float)
    weight = request.form.get("weight", type=float)

    if not file or not validate_metadata(age, gender, height, weight):
        return jsonify({"error": "Invalid input"}), 400

    filename = f"{test_id}_audio.wav"
    temp_path = os.path.join("/tmp", filename)
    file.save(temp_path)

    result = run_inference(temp_path)
    file_url = upload_to_gcs(temp_path, filename)
    os.remove(temp_path)

    return jsonify({
        "test_id": test_id,
        "prediction": result["label"],
        "confidence": round(result["confidence"], 2),
        "file_url": file_url,
        "metadata": {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight
        }
    })

@api.route("/delete/<test_id>", methods=["DELETE"])
def delete_record(test_id):
    filename = f"{test_id}_audio.wav"
    success = delete_from_gcs(filename)
    if success:
        return jsonify({"message": f"{test_id} deleted"}), 200
    return jsonify({"error": "File not found"}), 404
