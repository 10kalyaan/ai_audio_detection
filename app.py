from flask import Flask, request, jsonify
import os
import tempfile
import threading
from werkzeug.utils import secure_filename

# Import functionality from main.py
from main import (
    load_models_and_scalers,
    ensemble_predict,
    StreamingAudioEnsembleClassifier
)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load models on startup
print("Loading models...")
model1, scaler1, model2, scaler2, model3, scaler3 = load_models_and_scalers()
print("Models loaded successfully")

# Streaming classifiers for active sessions
streaming_sessions = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def predict_audio():
    """Single file prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            verdict = ensemble_predict(
                filepath, 
                model1, scaler1, 
                model2, scaler2, 
                model3, scaler3
            )
            
            # Get individual model predictions
            from main import predict_model1, predict_model2, predict_model3
            model1_pred = predict_model1(filepath, model1, scaler1)
            model2_pred = predict_model2(filepath, model2, scaler2)
            model3_pred = predict_model3(filepath, model3, scaler3)
            
            result = {
                'verdict': verdict,
                'model_predictions': {
                    'model1': model1_pred,
                    'model2': model2_pred,
                    'model3': model3_pred
                }
            }
            
            # Clean up the temp file
            os.remove(filepath)
            
            return jsonify(result)
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/stream/start', methods=['POST'])
def start_streaming():
    """Start streaming session"""
    # Generate a unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Create new classifier for this session
    classifier = StreamingAudioEnsembleClassifier(
        sample_rate=16000,
        chunk_size=1024,
        buffer_seconds=3
    )
    
    # Store in sessions dict
    streaming_sessions[session_id] = {
        'classifier': classifier,
        'predictions': [],
        'status': 'initialized',
        'created_at': threading.Event()
    }
    
    return jsonify({
        'session_id': session_id,
        'status': 'initialized',
        'message': 'Streaming session created. Upload audio files to this session.'
    })

@app.route('/api/stream/<session_id>/upload', methods=['POST'])
def upload_to_stream(session_id):
    """Upload audio file to an existing streaming session"""
    if session_id not in streaming_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    if streaming_sessions[session_id]['status'] not in ['initialized', 'processing']:
        return jsonify({'error': 'Session is not in a valid state for uploads'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get the classifier
            classifier = streaming_sessions[session_id]['classifier']
            
            # Update session status
            streaming_sessions[session_id]['status'] = 'processing'
            streaming_sessions[session_id]['file_path'] = filepath
            
            # Start processing in background
            def process_file():
                try:
                    classifier.start_audio_stream(use_microphone=False, audio_file=filepath)
                    classifier.prediction_complete.wait(timeout=60)  # Wait up to 60 seconds
                    classifier.stop_audio_stream()
                    
                    # Store results
                    streaming_sessions[session_id]['predictions'] = classifier.all_predictions
                    streaming_sessions[session_id]['status'] = 'completed'
                    
                    # Cleanup
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    streaming_sessions[session_id]['status'] = 'error'
                    streaming_sessions[session_id]['error'] = str(e)
                    if os.path.exists(filepath):
                        os.remove(filepath)
            
            # Start processing thread
            thread = threading.Thread(target=process_file)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'session_id': session_id,
                'status': 'processing',
                'message': 'Audio file uploaded and being processed'
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            streaming_sessions[session_id]['status'] = 'error'
            streaming_sessions[session_id]['error'] = str(e)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/stream/<session_id>/status', methods=['GET'])
def get_stream_status(session_id):
    """Get status of a streaming session"""
    if session_id not in streaming_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = streaming_sessions[session_id]
    
    # Prepare response
    response = {
        'session_id': session_id,
        'status': session['status']
    }
    
    # Add results if completed
    if session['status'] == 'completed' and session.get('predictions'):
        from collections import Counter
        prediction_counts = Counter(session['predictions'])
        
        if prediction_counts:
            majority_prediction = prediction_counts.most_common(1)[0][0]
            majority_percentage = (prediction_counts[majority_prediction] / len(session['predictions'])) * 100
            
            response['result'] = {
                'verdict': majority_prediction,
                'confidence': round(majority_percentage, 1),
                'window_predictions': dict(prediction_counts)
            }
    
    # Add error information if applicable
    if session['status'] == 'error' and 'error' in session:
        response['error'] = session['error']
    
    return jsonify(response)

@app.route('/api/stream/<session_id>', methods=['DELETE'])
def delete_stream_session(session_id):
    """Delete a streaming session and clean up resources"""
    if session_id not in streaming_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        # Clean up classifier if needed
        if streaming_sessions[session_id]['status'] == 'processing':
            classifier = streaming_sessions[session_id]['classifier']
            classifier.stop_audio_stream()
        
        # Clean up file if exists
        if 'file_path' in streaming_sessions[session_id]:
            file_path = streaming_sessions[session_id]['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from sessions
        del streaming_sessions[session_id]
        
        return jsonify({
            'message': f'Session {session_id} deleted successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Audio prediction API is running'
    })

# Add cleanup handler for expired sessions
@app.before_request
def cleanup_old_sessions():
    """Clean up old sessions (run before each request)"""
    import time
    current_time = time.time()
    session_timeout = 3600  # 1 hour
    
    sessions_to_remove = []
    for session_id, session in streaming_sessions.items():
        # Check if session is too old
        if hasattr(session, 'created_at') and (current_time - session['created_at']) > session_timeout:
            sessions_to_remove.append(session_id)
    
    # Clean up old sessions
    for session_id in sessions_to_remove:
        try:
            if streaming_sessions[session_id]['status'] == 'processing':
                classifier = streaming_sessions[session_id]['classifier']
                classifier.stop_audio_stream()
            
            if 'file_path' in streaming_sessions[session_id]:
                file_path = streaming_sessions[session_id]['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            del streaming_sessions[session_id]
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)