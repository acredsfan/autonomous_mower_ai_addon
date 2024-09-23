from flask import Flask, Response
import requests
import tensorflow as tf
from io import BytesIO
from PIL import Image, ImageDraw

app = Flask(__name__)

# Load your object detection model
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320/saved_model')


def detect_objects(image):
    # Preprocess image for TensorFlow detection model
    input_tensor = tf.convert_to_tensor([image])
    detections = model(input_tensor)
    return detections


def process_frame(frame):
    """Process frame, run detection, and draw bounding boxes."""
    image = Image.open(BytesIO(frame))  # Convert the byte stream to an image
    detections = detect_objects(image)

    draw = ImageDraw.Draw(image)

    for detection in detections['detection_boxes']:
        # Example bounding box drawing logic using PIL
        # You can convert normalized coordinates (0-1) to pixel coordinates here
        draw.rectangle(((x1, y1), (x2, y2)), outline="green", width=2)
        draw.text((x1, y1), 'Object', fill="green")

    # Convert processed image back to JPEG
    output = BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()


def stream_from_pi4():
    """Stream video from Pi 4 and process it."""
    stream_url = 'http://<PI4_IP>:8000/video_feed'  # Change <PI4_IP> to your Pi 4 IP
    response = requests.get(stream_url, stream=True)

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            break

        processed_frame = process_frame(chunk)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(stream_from_pi4(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


# Define the class labels for the autnomous robot mower environment
# category_index = {1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Animal', 5: 'Tree', 6: 'Grass', 7: 'Bush', 8: 'Pavement', 
#                   9: 'Road', 10: 'Sidewalk', 11: 'Building', 12: 'Sky', 13: 'Cloud', 14: 'Sun', 
#                   15: 'Fence', 16: 'Sign', 17: 'Pole', 18: 'Ledge', 19: 'Traffic Sign', 20: 'Vegetation', 
#                   21: 'Terrain', 22: 'Water', 23: 'Boat', 24: 'Wind Turbine', 25: 'Construction', 26: 'Truck', 
#                   27: 'Bus', 28: 'Train', 29: 'Motorcycle', 30: 'Airplane', 31: 'Helicopter', 32: 'UAV', 
#                   33: 'Drone', 34: 'Power Line', 35: 'Bridge', 36: 'Building', 37: 'Tunnel', 38: 'Archway', 39: 'Column', 
#                   40: 'Balcony', 41: 'Window', 42: 'Door', 43: 'Ditch', 44: 'Drop-off', 45: 'Stairs', 46: 'Ramp', 
#                   47: 'Parking', 48: 'Bench', 49: 'Chair', 50: 'Table', 51: 'Bed', 52: 'Fire Pit', 53: 'Mulch', 
#                   54: 'Bucket', 55: 'Hose', 56: 'Sprinkler', 57: 'Fountain', 58: 'Pond', 59: 'Pool', 60: 'Stream',
#                   61: 'Lake', 62: 'River', 63: 'Lawn Mower', 64: 'Wheelbarrow', 65: 'Garden Tools', 66: 'Garden Hose',
#                   67: 'Garden Furniture', 68: 'Garden Shed', 69: 'Garden Fence', 70: 'Garden Gate', 71: 'Garden Path',
#                   72: 'Ball', 73: 'Frisbee', 74: 'Kite', 75: 'Boomerang', 76: 'Exension Cord', 77: 'Cable', 78: 'Hose Reel'}