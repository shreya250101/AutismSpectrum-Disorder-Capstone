from flask import Flask, render_template,  request
import pandas as pd
import pickle
from typing import Tuple
import tensorflow as tf
import numpy as np

model3 = pickle.load(open("0-3.sav", "rb"))
model11 = pickle.load(open("4-11.sav", "rb"))
MODEL_PATH = 'image_model.tflite'

app = Flask(__name__)

def get_interpreter(model_path: str) -> Tuple:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict(image_path: str) -> int:
    interpreter, input_details, output_details = get_interpreter(MODEL_PATH)
    input_shape = input_details[0]['shape']
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, (input_shape[2], input_shape[2]))
    img = tf.expand_dims(img, axis=0)
    resized_img = tf.cast(img, dtype=tf.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], resized_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return np.argmax(results, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Three_year', methods=['GET', 'POST'])
def Three_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'middle eastern':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'White European':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'black':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'asian':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'south asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Native Indian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if df['etnicity'] == 'Others':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])	
        if df['etnicity'] == 'mixed':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if df['etnicity'] == 'Pacifica':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model3.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Eleven_year', methods=['GET', 'POST'])
def Eleven_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'Others':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Middle Eastern':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'White-European':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Black':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'South Asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])	
        if df['etnicity'] == 'Asian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Pasifika':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if df['etnicity'] == 'Turkish':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model11.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Image', methods=['GET', 'POST'])
def Image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        path = 'static/test/'+filename
        Index = predict(path)

        print(name)
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email, prediction=prediction, img='http://127.0.0.1:5000/'+path)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)