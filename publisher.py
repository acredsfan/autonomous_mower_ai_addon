# publisher.py
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("localhost", 1883, 60)
client.publish("mower/status", "Pi 5 connected and ready")
client.disconnect()