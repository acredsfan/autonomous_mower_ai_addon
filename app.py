from flask import Flask, Response, render_template
import threading
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from logger_config import LoggerConfigInfo
import time
import cv2
import socket
import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import ctypes
import struct


# Initialize logger
logging = LoggerConfigInfo().get_logger(__name__)

# Load environment variables from .env file
load_dotenv()
HEF_PATH = os.getenv('HEF_PATH')  # Path to your Hailo HEF model file
POSTPROCESS_SO_PATH = os.getenv('POSTPROCESS_SO_PATH')  # Path to post-processing .so file
PI4_IP = os.getenv('PI4_IP')  # IP address of the Pi 4 sending the video stream
PI4_PORT = os.getenv('PI4_PORT', '8000')  # Port number for the video stream

logging.info(f"HEF_PATH: {HEF_PATH}")
logging.info(f"POSTPROCESS_SO_PATH: {POSTPROCESS_SO_PATH}")
logging.info(f"Pi4_IP: {PI4_IP}")

app = Flask(__name__)

# Initialize GStreamer
Gst.init(None)

# Global variables to store the latest processed frame
latest_frame = None
frame_lock = threading.Lock()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', int(PI4_PORT)))

def check_hailo_device():
    """
    Check if the Hailo device is available on the Pi 5.
    Returns:
        bool: True if the Hailo device is found, False otherwise.
    """
    available_devices = hailo.Device.scan()
    if len(available_devices) == 0:
        logging.error("No Hailo devices found")
        return False
    logging.info(f"Found {len(available_devices)} Hailo devices")
    return True


#Get UDP frame and preprocess it for YoloV8
def receive_udp_frame(sock):
    data, _ = sock.recvfrom(65536)
    frame_size = struct.unpack('>L', data[:4])[0]
    frame_data = data[4:]
    while len(frame_data) < frame_size:
        data, _ = sock.recvfrom(65536)
        frame_data += data
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    return cv2.imdecode(frame, cv2.IMREAD_COLOR)


# Take frame and process it with Hailo
def process_frame(frame):
    """
    Process a frame using the Hailo device.
    Args:
        frame (np.ndarray): The input frame to process.
    Returns:
        np.ndarray: The processed frame.
    """
    # Load the HEF model
    hef = HEF(HEF_PATH)
    devices = Device.scan()
    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        height, width, channels = hef.get_input_vstream_infos()[0].shape

        # Preprocess the frame
        frame = cv2.resize(frame, (width, height))
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis=0)

        # Create input and output vstreams
        input_vstreams = InputVStreams.create_from_params(input_vstreams_params)
        output_vstreams = OutputVStreams.create_from_params(output_vstreams_params)

        # Set input vstreams data
        input_vstreams[0].set_data(frame)

        # Infer the frame
        target.infer(network_group, input_vstreams, output_vstreams, network_group_params)

        # Get the output vstreams data
        output_data = output_vstreams[0].get_data()
        output_data = output_data[0].transpose(1, 2, 0)
        output_data = (output_data * 255).astype(np.uint8)

        return output_data


def postprocess_frame(frame):
    """
    Post-process the frame using the compiled shared object file.
    Args:
        frame (np.ndarray): The input frame to post-process.
    Returns:
        np.ndarray: The post-processed frame.
    """
    # Load the shared object file
    postprocess = ctypes.CDLL(POSTPROCESS_SO_PATH)

    # Define the function signature
    postprocess.postprocess_frame.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    postprocess.postprocess_frame.restype = ctypes.POINTER(ctypes.c_uint8)

    # Get the frame data
    frame_data = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    frame_height, frame_width, frame_channels = frame.shape

    # Post-process the frame
    postprocessed_frame = postprocess.postprocess_frame(
        frame_data, frame_height, frame_width, frame_channels
    )

    # Convert the post-processed frame to a numpy array
    postprocessed_frame = np.ctypeslib.as_array(
        postprocessed_frame, shape=(frame_height, frame_width, frame_channels)
    )

    return postprocessed_frame


def draw_boxes(frame, boxes):
    """
    Draw bounding boxes on the frame.
    Args:
        frame (np.ndarray): The input frame to draw the boxes on.
        boxes (list): A list of bounding boxes to draw.
    Returns:
        np.ndarray: The frame with the bounding boxes drawn.
    """
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def detect_objects(frame):
    """
    Detect objects in the frame using the Hailo device.
    Args:
        frame (np.ndarray): The input frame to detect objects in.
    Returns:
        np.ndarray: The frame with the detected objects drawn.
    """
    processed_frame = process_frame(frame)
    postprocessed_frame = postprocess_frame(processed_frame)
    return postprocessed_frame


def depth_detection(frame):
    """
    Detect depth in the frame using the Hailo device.
    Args:
        frame (np.ndarray): The input frame to detect depth in.
    Returns:
        np.ndarray: The frame with the depth detected.
    """
    processed_frame = process_frame(frame)
    postprocessed_frame = postprocess_frame(processed_frame)
    return postprocessed_frame


def edge_detection(frame):
    """
    Detect edges in the frame using the Hailo device.
    Args:
        frame (np.ndarray): The input frame to detect edges in.
    Returns:
        np.ndarray: The frame with the edges detected.
    """
    processed_frame = process_frame(frame)
    postprocessed_frame = postprocess_frame(processed_frame)
    return postprocessed_frame


def ledge_detection(frame):
    """
    Detect ledges in the frame using the Hailo device.
    Args:
        frame (np.ndarray): The input frame to detect ledges in.
    Returns:
        np.ndarray: The frame with the ledges detected.
    """
    processed_frame = process_frame(frame)
    postprocessed_frame = postprocess_frame(processed_frame)
    return postprocessed_frame


# Stream video with overlays and detections back to the client
def gst_pipeline_thread():
    """
    Gstreamer pipeline to take processed frames and stream them to the client.
    """
    while True:
        # Receive frame from UDP stream
        frame = receive_udp_frame(sock)

        # Detect objects in the frame
        processed_frame = detect_objects(frame)

        # Draw bounding boxes on the frame
        processed_frame = draw_boxes(processed_frame, [[100, 100, 200, 200]])

        # Convert the frame to JPEG format
        _, jpeg_frame = cv2.imencode('.jpg', processed_frame)

        with frame_lock:
            global latest_frame
            latest_frame = jpeg_frame.tobytes()



@app.route('/video_feed')
def video_feed():
    """
    Flask route that generates the live video feed for the WebUI.
    This function streams the latest processed frame (JPEG format) to the client.
    Returns:
        Response: The video stream in multipart/x-mixed-replace format.
    """
    def generate():
        while True:
            with frame_lock:
                if latest_frame:
                    frame_data = latest_frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.03)  # Adjust as needed for frame rate
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    Flask route to render the index page (WebUI) for viewing the live stream.
    """
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """
    Health check route to verify if the Pi 5 is running and responsive.
    Returns:
        Response: HTTP 200 status with "OK".
    """
    return "OK", 200


if __name__ == "__main__":
    # Check if the Hailo device is available before starting the pipeline
    if check_hailo_device():
        # Start GStreamer pipeline in a separate thread
        gst_thread = threading.Thread(target=gst_pipeline_thread)
        gst_thread.daemon = True
        gst_thread.start()
        # Start the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logging.error("Exiting: Hailo device not found.")
