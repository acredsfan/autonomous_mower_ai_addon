from flask import Flask, Response, request, jsonify
import requests
import threading
import io
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import numpy as np
import hailo_platform.pyhailort as hailort
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
PI4_IP = os.getenv('PI4_IP')  # IP address of the Pi 4

app = Flask(__name__)

# Load the Hailo model
HEF_PATH = os.getenv('HEF_PATH')  # Path to your Hailo HEF model file
device = hailort.Device()
hef = hailort.HEF(HEF_PATH)
network_group = device.configure(hef)
input_vstream = network_group.create_input_vstream()
output_vstream = network_group.create_output_vstream()

def preprocess_image(image):
    """Preprocess the image for Hailo model."""
    image = image.resize((640, 640))  # Adjust to your model's expected input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def run_inference_on_hailo(image):
    """Run inference on Hailo chip."""
    preprocessed_image = preprocess_image(image)

    # Send the preprocessed image to the Hailo input stream
    input_vstream.write(preprocessed_image)

    # Get the output from the Hailo output stream
    output = output_vstream.read()

    # Interpret the output according to your model
    # This will depend on your model's output format
    predictions = interpret_output(output)
    return predictions

def interpret_output(output):
    """Interpret the model output and return predictions."""
    # Implement according to your model's output
    # For example:
    predictions = []
    # Parse output and populate predictions list
    return predictions

def process_frame(image):
    """Process each frame for object detection."""
    # Run inference
    predictions = run_inference_on_hailo(image)

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        label = pred['label']
        confidence = pred['confidence']
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1), f'{label} ({confidence:.2f})', fill="green")

    # Convert processed image back to bytes
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()

def stream_video():
    """Stream video from Pi 4, process it, and serve to clients."""
    stream_url = f'http://{PI4_IP}:8000/video_feed'
    response = requests.get(stream_url, stream=True)
    bytes_buffer = b''
    for chunk in response.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            image = Image.open(io.BytesIO(jpg))
            processed_frame = process_frame(image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    predictions = run_inference_on_hailo(image)
    # Determine if obstacle is detected based on your criteria
    obstacle_detected = any(pred['confidence'] > 0.5 for pred in predictions)
    return jsonify({'obstacle_detected': obstacle_detected})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
