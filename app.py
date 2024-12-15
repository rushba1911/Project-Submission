from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Recreate the model architecture
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)  # Match your number of classes

# Load state dictionary
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    raise Exception("Model file 'best_model.pth' not found. Ensure the file exists in the correct directory.")

# Set the model to evaluation mode
model.eval()

# Set up upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global variables to store stats and history
stats = {
    'totalClassifications': 0,
    'recyclableCount': 0,
    'nonRecyclableCount': 0,
}

history = []

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing (same transformations used during training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Route for uploading and predicting the image class
@app.route('/upload_image', methods=['POST'])
def predict():
    # Check if an image is part of the POST request
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    # If no file is selected or file is empty
    if file.filename == '' or not file:
        return jsonify({'error': 'No selected file or empty file'}), 400

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)  # Save the image to the 'uploads/' folder
    except Exception as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 500

    # Open the image file
    try:
        img = Image.open(file.stream).convert("RGB")  # Ensure image is in RGB mode
    except Exception as e:
        return jsonify({'error': f'Failed to open image: {str(e)}'}), 400

    # Preprocess the image
    try:
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(torch.device('cpu'))  # Ensure the tensor is on the correct device
    except Exception as e:
        return jsonify({'error': f'Failed to preprocess image: {str(e)}'}), 400

    # Make a prediction
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Get softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Custom threshold for "Recyclable" class (index 1)
            threshold = 0.4  # Adjust this value based on your model's behavior
            if probs[0][1] > threshold:  # If probability of "Recyclable" > threshold
                predicted_label = "Recyclable"
            else:
                predicted_label = "Non-Recyclable"
            
            # Extract confidence
            confidence = probs.max().item()
    except Exception as e:
        return jsonify({'error': f'Failed to predict: {str(e)}'}), 500
    
    # Update statistics
    stats['totalClassifications'] += 1
    if predicted_label == 'Recyclable':
        stats['recyclableCount'] += 1
    else:
        stats['nonRecyclableCount'] += 1

    # Add to history
    history.append({
        'id': stats['totalClassifications'],
        'category': predicted_label,
        'confidence': f"{confidence * 100:.2f}%",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_url': f'/uploads/{filename}'  # Include the path to the uploaded image
    })

    return jsonify({
        'category': predicted_label,
        'confidence': confidence,
        'image_url': f'/uploads/{filename}'  # Return the URL of the uploaded image
    })

# Route to get classification statistics
@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'totalClassifications': stats['totalClassifications'],
        'recyclablePercentage': (stats['recyclableCount'] / stats['totalClassifications']) * 100 if stats['totalClassifications'] > 0 else 0,
        'nonRecyclablePercentage': (stats['nonRecyclableCount'] / stats['totalClassifications']) * 100 if stats['totalClassifications'] > 0 else 0,
    })

# Route to get classification history
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history)

# Route for checking if the server is running
@app.route('/')
def home():
    return '''
        <html>
            <body>
                <h1>Model is ready for predictions</h1>
                <a href="/user">Go to User Page</a>
            </body>
        </html>
    '''

# Route to render user.html
@app.route('/user')
def user_page():
    return render_template('user.html')  # Assumes 'user.html' is in the 'templates' folder

# Serve the uploaded images from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask app
    app.run(debug=True)
