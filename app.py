from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'train_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODELS_FOLDER = 'models'
os.makedirs(MODELS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')
    class_name = request.form.get('className')

    if not class_name:
        return jsonify({'error': 'No class name provided'}), 400

    class_folder = os.path.join(UPLOAD_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)

    for file in files:
        file.save(os.path.join(class_folder, file.filename))

    return jsonify({'message': 'Files successfully uploaded'}), 200

@app.route('/train', methods=['POST'])
def train_model():
    try:
        subprocess.run(['python', 'training_script.py'], check=True)
        
        # Check if trained_model.h5 exists in models folder
        model_path = os.path.join(MODELS_FOLDER, 'trained_model.h5')
        if os.path.exists(model_path):
            return jsonify({'message': 'Training completed. Model ready for download'}), 200
        else:
            return jsonify({'error': 'Trained model not found'}), 404

    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model', methods=['GET'])
def download_model():
    model_path = os.path.join(MODELS_FOLDER, 'trained_model.h5')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Trained model not found'}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
