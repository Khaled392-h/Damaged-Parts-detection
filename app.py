import os
from flask import Flask, request, render_template, flash, send_from_directory, session
import cv2
import tensorflow as tf
import numpy as np
import atexit

global path


CATEGORIES = ['Not a car', 'Car']


carDetect = tf.keras.models.load_model('./models/carDetectionModel.model')
damageDetect = tf.keras.models.load_model('./models/damageDetection.model')
positionDetect = tf.keras.models.load_model('./models/positionDetection.model')


def prepare(path, img_size=256):
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


def car_detection():
    image = prepare('./uploads/' + file.filename)
    prediction = carDetect.predict([image])
    global carPer
    carPer = prediction[0][1] * 100
    carPer = '%.2f' % carPer
    if prediction[0][0] > prediction[0][1]:
        return False
    else:
        return True


def damage_detection():
    image = prepare('./uploads/' + file.filename)
    prediction = damageDetect.predict([image])
    global damagePer
    damagePer = prediction[0][0] * 100
    damagePer = '%.2f' % damagePer
    d1 = prediction[0][1] * 100
    print('%.2f' % d1, 'Not damaged')
    if prediction[0][0] > prediction[0][1]:
        return True
    else:
        return False


def position_detection():
    image = prepare('./uploads/' + file.filename, 224)
    prediction = positionDetect.predict([image])
    global position
    position = np.argmax(prediction)
    print("Position of car ", position)
    return position


UPLOAD_FOLDER = r'uploads/'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def a():
    path = 0
    if request.method == 'POST':
        print("post")
        if 'file' not in request.files:
            flash('No file part')

        global file

        file = request.files['file']
        if file.filename == '':
            print('no no no no')
            # flash('No File uploaded')
            return " No File uploaded" # render_template("index1.html")

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(path)
        file.save(path)
        print('Image saved')

        if car_detection():
            if damage_detection():
                cd = 'true'  # "It is a Car"
                dd = 'true'  # "It is Damaged"
                position = position_detection()
                if position == 0:
                    p = 'Front'
                elif position == 1:
                    p = 'Rear'
                else:
                    p = 'Side'

                return render_template('new result.html', cd=cd, dd=dd, path=path, fname=file.filename, carPer=position, damagePer=damagePer, p=p)
            else:
                cd = 'true'  # "It is a Car"
                nd = 'false'  # "It is Not Damaged"
                return render_template('new result.html', cd=cd, nd=nd, path=path, fname=file.filename, carPer=carPer, damagePer=damagePer)
        else:
            print('it is not a car')
            position_detection()
            return render_template('new result.html', nc="false", path=path, fname=file.filename, carPer=carPer, damagePer=0)

    print('**********')

    return render_template("indexNew.html", path=path)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contactUs.html')


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


if __name__ == '__main__':
    app.run()
