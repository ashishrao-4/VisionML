from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import subprocess
import firebase_admin
from firebase_admin import credentials, storage
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("visionml-flask-firebase-adminsdk-njze6-b90ca009af.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'visionml-flask.appspot.com'
})

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

    bucket = storage.bucket()

    for file in files:
        filename = secure_filename(file.filename)

        # Upload to Firebase Storage
        blob = bucket.blob(f"train_images/{class_name}/{filename}")
        blob.upload_from_file(file)

    return jsonify({'message': 'Files successfully uploaded'}), 200

@app.route('/train', methods=['POST'])
def train_model():
    try:
        subprocess.run(['python', 'training_script.py'], check=True)

        # Upload trained model to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob('models/trained_model.h5')
        blob.upload_from_filename('models/trained_model.h5')

        return jsonify({'message': 'Training completed. Model ready for download'}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model', methods=['GET'])
def download_model():
    bucket = storage.bucket()
    blob = bucket.blob('models/trained_model.h5')
    if blob.exists():
        model_path = 'models/trained_model.h5'
        blob.download_to_filename(model_path)
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Trained model not found'}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
