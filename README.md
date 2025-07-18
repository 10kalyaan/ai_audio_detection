# Real-Time Audio Deepfake Detection (Ensemble-Based)

This project is a real-time **deepfake audio detector** using an **ensemble of three ML models**. It supports both **microphone input** and **pre-recorded audio**. The ensemble method combines predictions from different models using unique features to deliver a robust verdict on whether the audio is real or fake.

---

## How It Works

The system uses **three separate models** trained on different feature sets and audio durations:

| Model | Input Duration | Features Used                         | Model Type |
|-------|----------------|----------------------------------------|------------|
| 1     | 10 seconds     | Spectral features, ZCR, RMS, etc.     | Scikit-learn (joblib) |
| 2     | 2 seconds      | MFCC, Chroma, ZCR, RMS                | Keras (.h5) |
| 3     | MFCC snapshot  | MFCC only                             | Keras (.h5) |

An **ensemble predictor** takes the results from all models. If **any** model flags the audio as `fake`, the final result is **FAKE**.

---

## Model Training

This project uses an ensemble of three audio classification models, each trained on different audio preprocessing strategies:

- **2-Second Model (`2sec.ipynb`)**  
  Trains on audio clips trimmed or padded to 2 seconds. Ideal for quick detection where speed is prioritized.
  
- **10-Second Model (`10sec.ipynb`)**  
  Trains on longer 10-second clips for better context and improved accuracy in scenarios where latency is less critical.

- **MFCC Model (`mfcc.ipynb`)**  
  Uses Mel-Frequency Cepstral Coefficients (MFCCs) extracted from the audio to train a model on speech-specific features.

These models are later ensembled to achieve more robust classification results across different audio lengths and characteristics.

### Dataset

All models are trained on the **Fake or Real Dataset** available on Kaggle:

[The Fake or Real Dataset – Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/)

---

## File Structure

```bash
.
├── main.ipynb             # Main notebook (contains all code)
├── models/
│   ├── 10sec.pkl          # Model 1
│   ├── 10sec_scaler.pkl
│   ├── 2sec.h5            # Model 2
│   ├── 2sec_scaler.pkl
│   ├── mfcc.h5            # Model 3
│   ├── mfcc_scaler.pkl
```

---

## Features

- Real-time audio streaming with PyAudio
- Fake detection from **microphone or audio file**
- Dynamic preprocessing: normalization, noise reduction, high-pass filtering
- Ensemble prediction for higher accuracy
- Live prediction output and final majority vote

---

## Requirements

Install dependencies using:

```bash
pip install numpy tensorflow librosa joblib pyaudio soundfile pandas scipy
```

> Note: On some platforms, PyAudio may need portaudio installed:
> 
> `sudo apt install portaudio19-dev` (Ubuntu) or use conda to install `pyaudio`.

---

## Usage

### 1. Run Detection From Audio File

```python
from main import StreamingAudioEnsembleClassifier

clf = StreamingAudioEnsembleClassifier()
clf.start_audio_stream(use_microphone=False, audio_file='your_audio.wav')
```

### 2. Run Real-Time Detection From Microphone

```python
clf = StreamingAudioEnsembleClassifier()
clf.start_audio_stream(use_microphone=True)
```

### 3. Stop Streaming

```python
clf.stop_audio_stream()
```

---

## Output Example

```
🔍 Predicting for file: temp_stream.wav
Model Predictions: ['real', 'fake', 'real']

🟥 FINAL VERDICT: FAKE

FINAL ENSEMBLE PREDICTION RESULTS
Total windows analyzed: 24
Distribution of ensemble predictions: {'REAL': 8, 'FAKE': 16}
FINAL VERDICT: This audio is most likely FAKE (66.7% of the windows agreed)
```

---

## Notes

- Models must be placed inside the `./models/` folder.
- Designed for binary classification: `real` vs `fake`.

## 🙌 Acknowledgements

Thanks to the open-source community and datasets enabling research in audio deepfake detection.  
Built with ❤️ using TensorFlow, Librosa, PyAudio, and more.
