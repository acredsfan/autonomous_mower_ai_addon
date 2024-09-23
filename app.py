from flask import Flask, Response
import requests
import tensorflow as tf
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model for obstacle, surface, and drop-off detection
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
mobilenet_model.trainable = False  # Freeze the layers for inference

# Load a separate model for edge detection (to identify drop-offs)
# This model can be a custom-trained model or a pre-trained model that identifies sharp edges
dropoff_model = tf.saved_model.load('dropoff_detection_model')  # Placeholder for drop-off detection model

# Map for object classes from ImageNet or a custom set of classes based on your dataset
CLASS_LABELS = {
    1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Animal', 5: 'Tree', 6: 'Grass', 7: 'Bush', 8: 'Pavement', 
    9: 'Road', 10: 'Sidewalk', 11: 'Building', 12: 'Sky', 13: 'Cloud', 14: 'Sun', 
    15: 'Fence', 16: 'Sign', 17: 'Pole', 18: 'Ledge', 19: 'Traffic Sign', 20: 'Vegetation', 
    21: 'Terrain', 22: 'Water', 23: 'Boat', 24: 'Wind Turbine', 25: 'Construction', 26: 'Truck', 
    27: 'Bus', 28: 'Train', 29: 'Motorcycle', 30: 'Airplane', 31: 'Helicopter', 32: 'UAV', 
    33: 'Drone', 34: 'Power Line', 35: 'Bridge', 36: 'Building', 37: 'Tunnel', 38: 'Archway', 39: 'Column', 
    40: 'Balcony', 41: 'Window', 42: 'Door', 43: 'Ditch', 44: 'Drop-off', 45: 'Stairs', 46: 'Ramp', 
    47: 'Parking', 48: 'Bench', 49: 'Chair', 50: 'Table', 51: 'Bed', 52: 'Fire Pit', 53: 'Mulch', 
    54: 'Bucket', 55: 'Hose', 56: 'Sprinkler', 57: 'Fountain', 58: 'Pond', 59: 'Pool', 60: 'Stream',
    61: 'Lake', 62: 'River', 63: 'Lawn Mower', 64: 'Wheelbarrow', 65: 'Garden Tools', 66: 'Garden Hose',
    67: 'Garden Furniture', 68: 'Garden Shed', 69: 'Garden Fence', 70: 'Garden Gate', 71: 'Garden Path',
    72: 'Ball', 73: 'Frisbee', 74: 'Kite', 75: 'Boomerang', 76: 'Exension Cord', 77: 'Cable', 78: 'Hose Reel'}


def preprocess_image(image):
    """Preprocess the image for MobileNetV2 model."""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)
    return image


def detect_objects_and_surfaces(image):
    """Run detection for both obstacles and surfaces using MobileNetV2."""
    preprocessed_image = preprocess_image(image)
    
    # Get predictions from the MobileNetV2 model
    predictions = mobilenet_model.predict(preprocessed_image)
    top_pred_idx = np.argmax(predictions[0])  # Get the class with the highest score
    confidence = predictions[0][top_pred_idx]
    
    # Get the corresponding label from CLASS_LABELS
    predicted_label = CLASS_LABELS.get(top_pred_idx, 'Unknown')
    return predicted_label, confidence


def detect_dropoff(image):
    """Run drop-off detection using an edge detection or depth model."""
    # Process image through the drop-off detection model
    preprocessed_image = preprocess_image(image)
    dropoff_detected = dropoff_model(preprocessed_image)
    
    # Interpret the result as a boolean indicating drop-off presence
    return dropoff_detected > 0.5  # Threshold can be adjusted


def process_frame(frame):
    """Process each frame for object detection, surface classification, and drop-off detection."""
    image = Image.open(BytesIO(frame))  # Convert the byte stream to an image
    
    # Detect objects and surfaces
    detected_label, confidence = detect_objects_and_surfaces(image)
    
    # Detect drop-offs or ditches
    dropoff_detected = detect_dropoff(image)
    
    # Drawing bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    
    # Draw label for the detected object or surface
    draw.text((10, 10), f'Detected: {detected_label} (Conf: {confidence:.2f})', fill="green")
    
    # Draw drop-off warning if detected
    if dropoff_detected:
        draw.text((10, 50), 'Warning: Drop-off detected!', fill="red")
    
    # Convert processed image back to JPEG
    output = BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()


def stream_from_pi4():
    """Stream video from Pi 4 and process it on Pi 5."""
    stream_url = 'http://<PI4_IP>:8000/video_feed'  # Change <PI4_IP> to the Pi 4 IP
    response = requests.get(stream_url, stream=True)

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            break
        
        # Process the current frame
        processed_frame = process_frame(chunk)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream_from_pi4(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
