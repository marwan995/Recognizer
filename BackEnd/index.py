from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PLATES_FILE'] = 'uploads/plates.txt'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable as a set to store unique plates
global_plates = set()

def plate_extractor(image_path):
    print(image_path)
    return image_path

def save_plate(license_plate):
    global global_plates
    if license_plate not in global_plates:
        global_plates.add(license_plate)

        # Also save to the file for persistence
        with open(app.config['PLATES_FILE'], 'a') as file:
            file.write(license_plate + '\n')

def read_plates():
    plates = set()
    if os.path.exists(app.config['PLATES_FILE']):
        with open(app.config['PLATES_FILE'], 'r') as file:
            plates = {line.strip() for line in file}
    return plates

@app.route('/', methods=['GET'])
def ping():
    # Retrieve plates from the global variable
    existing_plates = list(global_plates)
    # Return the plates in the response
    return jsonify({'message': 'Pong! The Flask app is running.', 'plates': existing_plates})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_class = plate_extractor(file_path)

        return jsonify({'license': len(predicted_class), 'canPass': True})

@app.route('/add_plate', methods=['POST'])
def add_plate():
    data = request.get_json()
    if 'license_plate' not in data:
        return jsonify({'error': 'License plate not provided'})

    license_plate = data['license_plate']
    save_plate(license_plate)

    return jsonify({'message': 'License plate added successfully'})

if __name__ == '__main__':
    # Initialize global plates from the file when the server starts
    global_plates = read_plates()
    print('Existing Plates:', global_plates)

    app.run()
