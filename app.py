# app.py
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt

app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to store the latest data from the mower
mower_data = {"status": "Disconnected", "obstacle": None, "battery": None}

# MQTT Client Setup
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe("mower/status")
    client.subscribe("mower/obstacle")
    client.subscribe("mower/battery")

def on_message(client, userdata, msg):
    global mower_data
    topic = msg.topic
    payload = msg.payload.decode()

    # Update mower data based on the received topic
    if topic == "mower/status":
        mower_data["status"] = payload
    elif topic == "mower/obstacle":
        mower_data["obstacle"] = payload
    elif topic == "mower/battery":
        mower_data["battery"] = payload

    # Emit updated data to the web interface
    socketio.emit('update_data', mower_data)

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60)  # Update with the correct IP if running MQTT broker elsewhere
mqtt_client.loop_start()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
