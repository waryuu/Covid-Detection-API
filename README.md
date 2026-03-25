# Covid-Detection-API

A Machine Learning-powered REST API designed to detect COVID-19 indicators from audio recordings (coughs/breath). This project bridges the gap between AI research and mobile health applications by providing a scalable inference engine on the cloud.

## 🌟 Overview
This API serves as the backend for a mobile application. It processes raw audio files, extracts key features using Digital Signal Processing (DSP), and runs them through a specialized **TensorFlow Lite** model to provide real-time health insights.

## 🛠️ Tech Stack
- **Framework:** Flask (Python)
- **Machine Learning:** TensorFlow Lite (Inference)
- **Audio Processing:** Librosa
- **Cloud Infrastructure:** Google App Engine & Google Cloud Storage (GCS)

---

## 🚀 How to Run

### 1. Local Setup
Ensure you have Python 3.9 installed, then follow these steps:

```bash
# Clone the repository
git clone [https://github.com/waryuu/Covid-Detection-API.git](https://github.com/waryuu/Covid-Detection-API.git)
cd Covid-Detection-API

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

### 2. Deployment (Google Cloud Platform)
This project is pre-configured for Google App Engine. To deploy to the web:

```bash
# Initialize gcloud CLI
gcloud init

# Deploy to App Engine
gcloud app deploy
```


## 📂 Project Structure

- app/routes.py: API endpoints and input validation.

- app/ml_service.py: Audio preprocessing and TFLite model execution.

- app/utils.py: Google Cloud Storage integration.

- app/model/: Directory for the .tflite model file.

Developed as part of a collaborative effort between Machine Learning, Mobile, and Cloud Computing teams.