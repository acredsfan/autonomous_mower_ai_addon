# app.py - Pi 5

from flask import Flask, Response
import threading
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from logger_config import LoggerConfigInfo
import time

# Initialize logger
logging = LoggerConfigInfo().get_logger(__name__)

# Load environment variables from .env file
load_dotenv()
HEF_PATH = os.getenv('HEF_PATH')  # Path to your Hailo HEF model file
POSTPROCESS_SO_PATH = os.getenv('POSTPROCESS_SO_PATH')  # Path to your post-processing .so file

app = Flask(__name__)

# Initialize GStreamer
Gst.init(None)

# Global variables to store the latest processed frame
latest_frame = None
frame_lock = threading.Lock()

def on_new_sample(sink, data):
    """Callback function called when a new sample is ready from the appsink."""
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
    """Function that runs the GStreamer pipeline."""
    pipeline_str = f"""
        udpsrc port=5000 caps = "application/x-rtp, media=video, encoding-name=H264, payload=96" ! 
        rtph264depay ! 
        h264parse ! 
        avdec_h264 ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,format=RGB,width=640,height=640 !
        tee name=t !
        hailomuxer name=hmux !
        hailonet hef-path={HEF_PATH} batch-size=1 force-writable=true ! 
        hailofilter so-path={POSTPROCESS_SO_PATH} qos=false ! 
        hailooverlay ! 
        videoconvert n-threads=3 qos=false ! 
        jpegenc ! 
        appsink name=appsink emit-signals=true max-buffers=1 drop=true
    """
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('appsink')
    appsink.connect('new-sample', on_new_sample, None)
    appsink.set_property('emit-signals', True)
    appsink.set_property('max-buffers', 1)
    appsink.set_property('drop', True)
    
    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # Run the pipeline
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
    pipeline.set_state(Gst.State.NULL)

@app.route('/video_feed')
def video_feed():
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
    return render_template('index.html')

if __name__ == "__main__":
    # Start GStreamer pipeline in a separate thread
    gst_thread = threading.Thread(target=gst_pipeline_thread)
    gst_thread.daemon = True
    gst_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
