import os
import numpy as np
import tensorflow as tf
import librosa

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "covid_model.tflite")
)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    target_len = 16000 * 3  # 3 seconds audio length
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    return np.expand_dims(y, axis=0).astype(np.float32)

def run_inference(file_path):
    input_data = preprocess_audio(file_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    label = "positif" if prediction[0] > 0.5 else "negatif"
    return {
        "label": label,
        "confidence": float(prediction[0], 2)
    }
