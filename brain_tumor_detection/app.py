from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.detector import analyze_mri

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or similar image formats.'}), 400

    image_bytes = file.read()
    result = analyze_mri(image_bytes)

    if 'error' in result:
        return jsonify(result), 400

    return jsonify(result)

if __name__ == '__main__':
    print("🧠 Brain Tumor Detection System")
    print("   Running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
