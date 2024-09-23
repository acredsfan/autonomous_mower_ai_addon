import cv2
import socket
import struct
import pickle
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load pre-trained TensorFlow model (you can change the path to the model you're using)
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320/saved_model')  # Update the path if needed
# Define the class labels for the autnomous robot mower environment
category_index = {1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Animal', 5: 'Tree', 6: 'Grass', 7: 'Bush', 8: 'Pavement', 
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


def detect_objects(frame):
    # Preprocess the frame and run inference
    input_tensor = tf.convert_to_tensor([frame])
    detections = model(input_tensor)
    # Post-process to extract bounding boxes and class IDs
    return detections

def video_stream():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('PiMowBot.local', 8080))  # Replace with your Pi 4 IP
    data = b""
    payload_size = struct.calcsize("L")

    while True:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # Perform object detection
        detections = detect_objects(frame)

        # Draw bounding boxes and labels on the frame
        for detection in detections['detection_boxes']:
            # Extract box coordinates and class label, confidence score
            # Add drawing code here

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

