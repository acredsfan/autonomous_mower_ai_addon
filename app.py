# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "Mower Control Interface is up!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)