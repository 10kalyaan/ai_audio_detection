import os
import numpy as np
import tensorflow as tf
import librosa
import joblib
import pyaudio
import time
import queue
import threading
import wave
import soundfile as sf
from collections import Counter
import pandas as pd

# MODEL 1 SETUP (10sec)
MODEL1_PATH = "./models/10sec.pkl"
SCALER1_PATH = "./models/10sec_scaler.pkl"
SAMPLE_RATE_1 = 22050

def extract_features_model1(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE_1)
        features = {}
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        y_harmonic = librosa.effects.harmonic(y)
        y_percussive = librosa.effects.percussive(y)
        features['harmonic_mean'] = np.mean(y_harmonic)
        features['harmonic_std'] = np.std(y_harmonic)
        features['percussive_mean'] = np.mean(y_percussive)
        features['percussive_std'] = np.std(y_percussive)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_dynamic_range'] = np.max(rms) - np.min(rms)
        return pd.DataFrame([features])
    except Exception as e:
        print(f"[Model 1] Error processing: {e}")
        return None

def predict_model1(file_path, model, scaler):
    features = extract_features_model1(file_path)
    if features is None:
        return None
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    return 'fake' if pred[0] == 1 else 'real'

# MODEL 2 SETUP (2sec)
MODEL2_PATH = "./models/2sec.h5"
SCALER2_PATH = "./models/2sec_scaler.pkl"
SAMPLE_RATE_2 = 16000

def extract_features_model2(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE_2)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        features = np.concatenate([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(spectral_centroids.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(rms.T, axis=0)
        ])
        return features
    except Exception as e:
        print(f"[Model 2] Error processing: {e}")
        return None

def predict_model2(file_path, model, scaler):
    features = extract_features_model2(file_path)
    if features is None:
        return None
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0][0]
    return 'fake' if prediction > 0.5 else 'real'

# MODEL 3 SETUP (MFCC)
MODEL3_PATH = "./models/mfcc.h5"
SCALER3_PATH = "./models/mfcc_scaler.pkl"

def extract_features_model3(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"[Model 3] Error processing: {e}")
        return None

def predict_model3(file_path, model, scaler):
    features = extract_features_model3(file_path)
    if features is None:
        return None
    scaled = scaler.transform([features])
    pred = model.predict(scaled)
    label = np.argmax(pred)
    return 'fake' if label == 1 else 'real'

def load_models_and_scalers():
    model1 = joblib.load(MODEL1_PATH)
    scaler1 = joblib.load(SCALER1_PATH)

    model2 = tf.keras.models.load_model(MODEL2_PATH)
    scaler2 = joblib.load(SCALER2_PATH)

    model3 = tf.keras.models.load_model(MODEL3_PATH)
    scaler3 = joblib.load(SCALER3_PATH)

    return model1, scaler1, model2, scaler2, model3, scaler3

def ensemble_predict(file_path, model1, scaler1, model2, scaler2, model3, scaler3):
    print(f"\n🔍 Predicting for file: {file_path}")
    
    predictions = []
    predictions.append(predict_model1(file_path, model1, scaler1))
    predictions.append(predict_model2(file_path, model2, scaler2))
    predictions.append(predict_model3(file_path, model3, scaler3))

    print(f"Model Predictions: {predictions}")

    # FAKE if any model predicts FAKE
    final_verdict = 'FAKE' if "fake" in predictions else 'REAL'

    if final_verdict == 'FAKE':
        print(f"\n🟥 FINAL VERDICT: {final_verdict}")
    else:
        print(f"\n🟩 FINAL VERDICT: {final_verdict}")
    
    return final_verdict

class StreamingAudioEnsembleClassifier:
    def __init__(self, sample_rate=16000, chunk_size=1024, buffer_seconds=3):
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_size = self.sample_rate * self.buffer_seconds
        
        # Audio buffer for accumulating data
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # For overlapping windows
        self.overlap_factor = 0.5  # 50% overlap
        self.step_size = int(self.buffer_size * (1 - self.overlap_factor))
        
        self.all_predictions = []  # ensemble verdicts for each window
        self.prediction_complete = threading.Event()
        
        # Audio stream setup
        self.audio_queue = queue.Queue()
        self.stop_flag = False
        
        # Debug flag
        self.debug = True
        
        # Preprocessing flags (if desired)
        self.normalize_audio = True
        self.apply_noise_reduction = True
        
        (self.model1, self.scaler1,
         self.model2, self.scaler2,
         self.model3, self.scaler3) = load_models_and_scalers()
        
        if self.debug:
            print("Ensemble models loaded successfully.")
    
    def start_audio_stream(self, use_microphone=True, audio_file=None):
        """Start capturing audio from microphone or file in a separate thread"""
        self.stop_flag = False
        self.all_predictions = []
        self.prediction_complete.clear()
        
        if use_microphone:
            self._start_microphone_stream()
        elif audio_file:
            self._start_file_stream(audio_file)
        else:
            raise ValueError("Either use_microphone must be True or audio_file must be provided")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        if use_microphone:
            print("Audio streaming started from microphone. Speak now...")
        else:
            print(f"Audio streaming started from file: {audio_file}")
    
    def _start_microphone_stream(self):
        """Start streaming from microphone"""
        self.p = pyaudio.PyAudio()
        device_info = self.p.get_default_input_device_info()
        if self.debug:
            print(f"Using input device: {device_info['name']}")
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
    
    def _audio_callback(self, in_data):
        """Callback for microphone streaming"""
        if not self.stop_flag:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_data = self._preprocess_audio(audio_data)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _preprocess_audio(self, audio_data):
        """Enhanced preprocessing for microphone input"""
        audio_data = audio_data * 3.0
        
        # Normalize 
        if self.normalize_audio:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0.01:
                audio_data = audio_data / max_val
        
        # Better noise reduction with a dynamic threshold
        if self.apply_noise_reduction:
            noise_floor = np.percentile(np.abs(audio_data), 15) * 2.5
            audio_data = np.where(np.abs(audio_data) < noise_floor, 0, audio_data)
        
        # Apply a simple high-pass filter to remove low-frequency noise
        if len(audio_data) > 10:
            b = [0.98, -0.98]  # Simple high-pass filter coefficients
            a = [1, -0.97]
            from scipy import signal
            audio_data = signal.lfilter(b, a, audio_data)
        
        return audio_data

    def _start_file_stream(self, audio_file):
        """Stream audio from a file in chunks to simulate real-time"""
        self.file_thread = threading.Thread(
            target=self._stream_from_file,
            args=(audio_file,)
        )
        self.file_thread.daemon = True
        self.file_thread.start()
    
    def _stream_from_file(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            total_samples = len(y)
            chunks = total_samples // self.chunk_size
            print(f"Streaming file in {chunks} chunks...")
            
            for i in range(chunks):
                if self.stop_flag:
                    break
                start = i * self.chunk_size
                end = start + self.chunk_size
                chunk = y[start:end]
                self.audio_queue.put(chunk)
                time.sleep(self.chunk_size / self.sample_rate * 0.5)
            
            if not self.stop_flag and total_samples % self.chunk_size > 0:
                start = chunks * self.chunk_size
                chunk = y[start:]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                self.audio_queue.put(chunk)
            
            print("File streaming completed, waiting for final processing...")
            time.sleep(self.buffer_seconds * 2)
            self.prediction_complete.set()
        except Exception as e:
            print(f"Error streaming from file: {e}")
            self.prediction_complete.set()
    
    def _process_audio(self):
        """Process incoming audio chunks and make ensemble predictions"""
        prediction_count = 0
        last_buffer_energy = 0
        
        while not self.stop_flag:
            try:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.prediction_complete.is_set():
                        break
                    continue
                
                shift = min(len(audio_chunk), self.step_size)
                self.audio_buffer = np.roll(self.audio_buffer, -shift)
                self.audio_buffer[-shift:] = audio_chunk[:shift]
                
                current_energy = np.mean(np.square(self.audio_buffer))
                if current_energy < 0.0001:
                    if last_buffer_energy >= 0.0001 and self.debug:
                        print("Silence detected, skipping ensemble prediction")
                    last_buffer_energy = current_energy
                    continue
                last_buffer_energy = current_energy
                
                prediction_count += 1

                if prediction_count >= 10:
                    self._ensemble_predict()
                    prediction_count = 0
                
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in audio processing: {e}")
        
        if not self.stop_flag:
            self._show_final_prediction()
    
    def _ensemble_predict(self):
        """Write current audio buffer to a temporary WAV file and run ensemble predictor"""
        try:
            temp_filename = "temp_stream.wav"

            sf.write(temp_filename, self.audio_buffer, self.sample_rate)

            verdict = ensemble_predict(temp_filename,
                                       self.model1, self.scaler1,
                                       self.model2, self.scaler2,
                                       self.model3, self.scaler3)
            self.all_predictions.append(verdict.upper())
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error during ensemble prediction: {e}")
    
    def _show_final_prediction(self):
        """Aggregate and display the ensemble predictions obtained over the streaming period"""
        if not self.all_predictions:
            print("No ensemble predictions were made.")
            return
        
        # Majority voting aggregation
        prediction_counts = Counter(self.all_predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]
        majority_percentage = (prediction_counts[majority_prediction] / len(self.all_predictions)) * 100
        
        print("\n" + "="*50)
        print("FINAL ENSEMBLE PREDICTION RESULTS")
        print("="*50)
        print(f"Total windows analyzed: {len(self.all_predictions)}")
        print(f"Distribution of ensemble predictions: {dict(prediction_counts)}")
        print(f"\nFINAL VERDICT: This audio is most likely {majority_prediction}")
        print(f"({majority_percentage:.1f}% of the windows agreed on this verdict)")
        print("="*50)
    
    def stop_audio_stream(self):
        """Stop the audio stream and clean up resources"""
        self.stop_flag = True
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1)
        if hasattr(self, 'file_thread') and getattr(self, 'file_thread', None) is not None:
            self.file_thread.join(timeout=1)
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("Audio streaming stopped.")
        self._show_final_prediction()


if __name__ == "__main__":
    try:
        classifier = StreamingAudioEnsembleClassifier(
            sample_rate=16000,
            chunk_size=1024,
            buffer_seconds=3
        )
        
        # Set to True to use microphone or provide an audio file path
        use_microphone = False
        audio_file = "./audio_files/anish.wav"
        
        if not use_microphone and not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found.")
            exit(1)
        
        classifier.start_audio_stream(
            use_microphone=use_microphone,
            audio_file=audio_file if not use_microphone else None
        )
        
        try:
            if not use_microphone:
                classifier.prediction_complete.wait()
                time.sleep(1)
            else:
                recording_duration = 30
                print(f"Listening for {recording_duration} seconds (press Ctrl+C to stop)...")
                for i in range(recording_duration):
                    if i % 5 == 0 and i > 0:
                        print(f"Listened for {i} seconds...")
                    time.sleep(1)
                print("Recording complete, processing final results...")
                time.sleep(2)
                classifier.prediction_complete.set()
                time.sleep(1)
        except KeyboardInterrupt:
            print("User interrupt, stopping...")
        finally:
            classifier.stop_audio_stream()
            
    except Exception as e:
        print(f"Error: {e}")
