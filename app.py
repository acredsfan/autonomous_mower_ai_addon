from flask import Flask, Response, render_template
import threading
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from logger_config import LoggerConfigInfo
import time
import hailo_platform as hailo

# Initialize logger
logging = LoggerConfigInfo().get_logger(__name__)

# Load environment variables from .env file
load_dotenv()
HEF_PATH = os.getenv('HEF_PATH')  # Path to your Hailo HEF model file
POSTPROCESS_SO_PATH = os.getenv('POSTPROCESS_SO_PATH')  # Path to post-processing .so file
PI4_IP = os.getenv('Pi4_IP')  # IP address of the Pi 4 sending the video stream

logging.info(f"HEF_PATH: {HEF_PATH}")
logging.info(f"POSTPROCESS_SO_PATH: {POSTPROCESS_SO_PATH}")
logging.info(f"Pi4_IP: {PI4_IP}")

app = Flask(__name__)

# Initialize GStreamer
Gst.init(None)

# Global variables to store the latest processed frame
latest_frame = None
frame_lock = threading.Lock()

def check_hailo_device():
    """
    Check if the Hailo device is available on the Pi 5.
    Returns:
        bool: True if the Hailo device is found, False otherwise.
    """
    available_devices = hailo.PcieDevice.scan_devices()
    if len(available_devices) == 0:
        logging.error("No Hailo devices found")
        return False
    logging.info(f"Found {len(available_devices)} Hailo devices")
    return True


def on_new_sample(sink, data):
    """
    GStreamer callback function that processes a new sample (video frame).
    Extracts and stores the latest frame for streaming via Flask.
    Args:
        sink: The GStreamer sink element.
        data: Additional user data passed to the callback.
    Returns:
        Gst.FlowReturn.OK: To indicate successful processing of the frame.
    """
    sample = sink.emit('pull-sample')
    buf = sample.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)
    if success:
        frame_data = map_info.data
        with frame_lock:
            global latest_frame
            latest_frame = frame_data
        buf.unmap(map_info)
    return Gst.FlowReturn.OK


def gst_pipeline_thread():
    """
    The function to run the GStreamer pipeline that receives the UDP stream,
    processes it through the Hailo AI for obstacle detection, surface identification,
    and edge detection, and overlays the results on the video.
    """
    pipeline_str = f"""
        udpsrc address={PI4_IP} port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! 
        rtph264depay ! 
        h264parse ! 
        avdec_h264 ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,format=RGB,width=1280,height=720 ! 
        hailonet hef-path={HEF_PATH} batch-size=1 ! 
        hailofilter so-path={POSTPROCESS_SO_PATH} ! 
        hailooverlay ! 
        videoconvert ! 
        jpegenc ! 
        appsink name=appsink emit-signals=true max-buffers=1 drop=true
    """
    pipeline = Gst.parse_launch(pipeline_str)

    # Connect the appsink to the callback function to handle new frames
    appsink = pipeline.get_by_name('appsink')
    appsink.connect('new-sample', on_new_sample, None)
    appsink.set_property('emit-signals', True)
    appsink.set_property('max-buffers', 1)
    appsink.set_property('drop', True)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Handle GStreamer bus messages (e.g., errors, EOS)
    bus = pipeline.get_bus()
    while True:
        message = bus.timed_pop_filtered(10000, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                logging.error(f"GStreamer Error: {err}, {debug}")
                break
            elif message.type == Gst.MessageType.EOS:
                logging.info("GStreamer End of Stream")
                break
        time.sleep(0.1)
    
    # Set pipeline state to NULL when done
    pipeline.set_state(Gst.State.NULL)


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
