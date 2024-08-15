from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Concatenate
from scipy.special import inv_boxcox
from scipy.stats import boxcox
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DLinearModel(tf.keras.Model):
    def __init__(self, linear_units, dnn_units):
        super(DLinearModel, self).__init__()
        self.linear_layer = Dense(units=linear_units, activation=None)
        self.dnn_layers = [Dense(units, activation='relu') for units in dnn_units]
        self.final_layer = Dense(units=1, activation='linear')

    def call(self, inputs):
        linear_output = self.linear_layer(inputs)
        dnn_input = tf.concat([inputs, linear_output], axis=-1)
        dnn_output = dnn_input
        for layer in self.dnn_layers:
            dnn_output = layer(dnn_output)
        output = self.final_layer(dnn_output)
        return output

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

wheat_disease_model = load_model('Models/wheat_disease_model.h5')
seed_quality_model = load_model('Models/Seed_quality.h5')
seed_germination_model = YOLO('Models/Seed_Germination.pt')
weed_detection_model = load_model('Models/Weed_classification_model.h5')
solar_model = tf.saved_model.load('Models/best_model_DLinear')
air_quality_model = joblib.load('Models/rfc_model.joblib')

wheat_disease_classes = ["Brown_rust", "Healthy", "Smut", "Yellow_rust"]
seed_quality_classes = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans', 'skin-damaged soybeans', 'Spotted soybeans']
weed_detection_classes = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


lambdas = [2.5468742717577206, 1.8680814672464994, 0.5464690175831447]
solar_scaler = StandardScaler()

def transform_data(df, lambdas):
    transformed_df = df.copy()
    for i, column in enumerate(df.columns[:-1]):
        data = df[df[column] > 0][column]
        transformed_data = boxcox(data, lambdas[i % len(lambdas)])
        transformed_df.loc[df[column] > 0, column] = transformed_data
    return transformed_df

def inverse_transform_data(data, lambdas):
    reversed_data = data.copy()
    for i, column in enumerate(data.columns[:-1]):
        reversed_data[column] = inv_boxcox(data[column], lambdas[i % len(lambdas)])
    return reversed_data

def predict_solar_power(file_path):
    data = pd.read_csv(file_path)
    data.drop(['YEAR', 'Month', 'Day', 'Hour', 'Pressure', 'Precipitable Water', 'Wind Direction', 'Wind Speed'], axis=1, inplace=True)
    transformed_data = transform_data(data, lambdas)
    X = transformed_data.iloc[:, :-1].values
    X_scaled = solar_scaler.fit_transform(X)
    solar_model = tf.saved_model.load('best_model_DLinear')
    input_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    y_pred = solar_model(input_tensor).numpy().flatten()
    transformed_data['Predicted'] = y_pred
    inverse_transformed_data = inverse_transform_data(transformed_data, lambdas)
    y_pred_inverse = inverse_transformed_data['Predicted']
    result_df = data.copy()
    result_df['Predicted Power'] = y_pred_inverse

    return result_df

def predict_air_quality(file_path):
    new_data = pd.read_csv(file_path, encoding='latin1')
    new_data.drop(['City', 'Date'], axis=1, inplace=True)
    x_new = new_data.drop(['Zone'], axis=1)
    x_new.drop(['Unnamed: 0'], axis=1, inplace=True)
    scaler = StandardScaler()
    x_new_scaled = scaler.fit_transform(x_new)
    predictions = air_quality_model.predict(x_new_scaled)
    results = pd.DataFrame(x_new, columns=x_new.columns)
    results['Predicted_Zone'] = predictions
    return results


def predict_wheat_disease(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.
    prediction = wheat_disease_model.predict(img_array)
    index = np.argmax(prediction)
    return wheat_disease_classes[index]

def predict_seed_quality(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.
    prediction = seed_quality_model.predict(img_array)
    index = np.argmax(prediction)
    return seed_quality_classes[index]

def predict_weed_detection(img_path):
    image = cv2.imread(img_path)
    segmented_image, segmented_path = segmented(image, img_path)
    resized_image = cv2.resize(segmented_image, (224, 224))
    processed_image = resized_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = weed_detection_model.predict(processed_image)
    predicted_class = weed_detection_classes[np.argmax(prediction)]
    return predicted_class, segmented_path

def segmented(image, image_path):
    foto = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_foto = cv2.cvtColor(foto, cv2.COLOR_RGB2HSV)
    colormin = (25, 50, 50)
    colormax = (86, 255, 255)
    mask = cv2.inRange(hsv_foto, colormin, colormax)
    result = cv2.bitwise_and(foto, foto, mask=mask)
    pil_image = Image.fromarray(result)
    segmented_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented')
    os.makedirs(segmented_dir, exist_ok=True)
    save_path = os.path.join(segmented_dir, f"segmented_{os.path.basename(image_path)}")
    pil_image.save(save_path)
    return result, save_path

def predict_seed_germination(img_path):
    image = cv2.imread(img_path)
    results = seed_germination_model.predict(image)
    coordinates = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    labels = [seed_germination_model.names[int(cls_id)] for cls_id in class_ids]
    for i, coord in enumerate(coordinates):
        x1, y1, x2, y2 = coord
        label = labels[i]
        text_color = (0, 255, 255) if label == "Gereminated" else (0, 0, 255)
        display_label = "1" if label == "Gereminated" else "0"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, display_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    legend_text = "Germinated: 1, Non-Germinated: 0"
    cv2.putText(image, legend_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpeg')
    cv2.imwrite(result_path, image)
    return result_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/wheat_disease', methods=['GET', 'POST'])
def wheat_disease():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_wheat_disease(file_path)
            return render_template('wheat_disease.html', filename=filename, prediction=prediction)
    return render_template('wheat_disease.html')

@app.route('/seed_quality', methods=['GET', 'POST'])
def seed_quality():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_seed_quality(file_path)
            return render_template('seed_quality.html', filename=filename, prediction=prediction)
    return render_template('seed_quality.html')

@app.route('/seed_germination', methods=['GET', 'POST'])
def seed_germination():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result_path = predict_seed_germination(file_path)
            return render_template('seed_germination.html', filename='result.jpeg')
    return render_template('seed_germination.html')

@app.route('/solar_power', methods=['GET', 'POST'])
def solar_power():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predictions_df = predict_solar_power(file_path)
            predictions_html = predictions_df.to_html()
            return render_template('solar_power.html', predictions_html=predictions_html)
    return render_template('solar_power.html')

@app.route('/air_quality', methods=['GET', 'POST'])
def air_quality():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predictions_df = predict_air_quality(file_path)
            predictions_html = predictions_df.to_html()
            return render_template('air_quality.html', predictions_html=predictions_html)
    return render_template('air_quality.html')

@app.route('/weed_detection', methods=['GET', 'POST'])
def weed_detection():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction, segmented_path = predict_weed_detection(file_path)
            return render_template('weed_detection.html', filename=filename, prediction=prediction, segmented_path=segmented_path)
    return render_template('weed_detection.html')

if __name__ == '__main__':
    app.run(port=5002, debug=True)
