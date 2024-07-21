from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import io
import subprocess
import firebase_admin
from firebase_admin import credentials, storage
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("visionml-flask-firebase-adminsdk-njze6-bd9b7dd69d.json")
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
        blob = bucket.blob(f"train_images/{class_name}/{filename}")
        blob.upload_from_string(file.read(), content_type=file.content_type)

    return jsonify({'message': 'Files successfully uploaded'}), 200

@app.route('/train', methods=['POST'])
def train_model():
    try:
        subprocess.run(['python', 'training_script.py'], check=True)
        return jsonify({'message': 'Training completed. Model uploaded to Firebase and ready for download.'}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model', methods=['GET'])
def download_model():
    bucket = storage.bucket()
    blob = bucket.blob('models/trained_model.h5')

    if blob.exists():
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='trained_model.h5', mimetype='application/octet-stream')
    else:
        return jsonify({'error': 'Trained model not found'}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
