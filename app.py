from flask import Flask, Response, request, jsonify
import threading
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
PI4_IP = os.getenv('PI4_IP')  # IP address of the Pi 4
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
        souphttpsrc location=http://{PI4_IP}:8000/video_feed ! 
        jpegdec ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,format=RGB,width=640,height=640 ! 
        hailonet hef-path={HEF_PATH} batch-size=1 ! 
        hailofilter so-path={POSTPROCESS_SO_PATH} ! 
        hailooverlay ! 
        videoconvert ! 
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
    
    # Create a GLib MainLoop to run GStreamer
    loop = GLib.MainLoop()
    try:
        loop.run()
    except Exception as e:
        logging.error(f"GStreamer pipeline error: {e}")
    finally:
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
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Start GStreamer pipeline in a separate thread
    gst_thread = threading.Thread(target=gst_pipeline_thread)
    gst_thread.daemon = True
    gst_thread.start()
    app.run(host='0.0.0.0', port=5000)
