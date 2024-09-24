import hailo_platform.pyhailort as hailort
from PIL import Image, ImageDraw
from flask import Flask, Response
import requests
from io import BytesIO
import numpy as np
from dotenv import load_dotenv
import os
from logger_config import LoggerConfigDebug

LoggerConfigDebug.configure_logging()

app = Flask(__name__)

# Load environment variables from .env file in project_root directory
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)


# Load the Hailo YOLOv8 model
# Path to the Hailo HEF model file (project_root/models/yolov8_hailo_model.hef)
HEF_PATH = os.path.join(project_root, 'models', 'yolov8_hailo_model.hef')
device = hailort.Device()
hef = hailort.HEF(HEF_PATH)
network_group = device.configure(hef)
input_vstream = network_group.create_input_vstream()
output_vstream = network_group.create_output_vstream()


def preprocess_image(image):
    """Preprocess the image for Hailo model."""
    image = image.resize((640, 640))  # Resize image to YOLOv8 input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def run_inference_on_hailo(image):
    """Run YOLOv8 inference on Hailo chip."""
    preprocessed_image = preprocess_image(image)

    # Send the preprocessed image to the Hailo input stream
    input_vstream.write(preprocessed_image)

    # Get the output from the Hailo output stream
    output = output_vstream.read()

    # Interpret the output for YOLOv8 detections
    return output


def process_frame(frame):
    """Process each frame for YOLOv8 object detection."""
    image = Image.open(BytesIO(frame))  # Convert the byte stream to an image

    # Run YOLOv8 inference on Hailo
    predictions = run_inference_on_hailo(image)

    # Draw bounding boxes and labels on the image using the predictions
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        # Extract bounding box, label, and confidence from YOLOv8 output
        x1, y1, x2, y2 = pred['bbox']
        label = pred['label']
        confidence = pred['confidence']

        # Draw bounding box and label on the image
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1), f'{label} ({confidence:.2f})', fill="green")

    # Convert processed image back to JPEG
    output = BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()


def stream_from_pi4():
    """Stream video from Pi 4 and process it using Hailo chip on Pi 5."""
    # From .env Pi4_IP variable
    stream_url = os.getenv('Pi4_IP')
    response = requests.get(stream_url, stream=True)

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            break

        # Process each frame using YOLOv8 and Hailo
        processed_frame = process_frame(chunk)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               processed_frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(stream_from_pi4(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
