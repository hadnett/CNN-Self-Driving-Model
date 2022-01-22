import socketio
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()
app = Flask(__name__)

# speed_limit = 30


def img_preprocess(img):
    """
    Preprocesses the images for model training via cropping, adding blur, resizing and normalising.
    :param img: The image to be preprocesses for training.
    :return: The preprocesses image.
    """
    img = img[60:135, :, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (64, 64))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    result = model.predict(image)
    vector = np.vectorize(float)
    x = vector(result)
    steering_angle = x[0][0]
    throttle = x[0][1]
    throttle = float(throttle)
    steering_angle = float(steering_angle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('best_model_22.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)