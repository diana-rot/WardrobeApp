from __future__ import division, print_function
import os
import gc
import numpy as np
from flask import render_template, request, session, url_for
from flaskapp import app, login_required,redirect
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename, send_from_directory
import pymongo
import requests
from gridfs import GridFS
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from keras.preprocessing import image
import calendar
import base64
from bson import ObjectId
from datetime import datetime
from flask import jsonify, request, session

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

from flask import send_file

client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test


import cv2
from sklearn.cluster import KMeans
import imutils

# MODEL_PATH = 'my_second_model.h5'
MODEL_PATH = 'my_model_june.h5'
# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28))
    img_array = np.asarray(img)
    x = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    result = int(img_array[0][0][0])
    print(result)
    if result > 128:
        img = cv2.bitwise_not(x)
    else:
        img = x
    img = img / 255
    img = (np.expand_dims(img, 0))
    preds = model.predict(img)

    print(f"preds type: {type(preds)}, preds: {preds}")
    # predicting color
    return preds

def predict_color(img_path):
    clusters = 5
    img = cv2.imread(img_path)
    org_img = img.copy()
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))
    print('After Flattening shape --> ', flat_img.shape)

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')

    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)
    print("the colour in the db")
    print(p_and_c[1])
    block = np.ones((50, 50, 3), dtype='uint')
    plt.figure(figsize=(12, 8))
    for i in range(clusters):
        plt.subplot(1, clusters, i + 1)
        block[:] = p_and_c[i][1][::-1]  # we have done this to convert bgr(opencv) to rgb(matplotlib)
        plt.imshow(block)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(round(p_and_c[i][0] * 100, 2)) + '%')

    bar = np.ones((50, 500, 3), dtype='uint')
    plt.figure(figsize=(12, 8))
    plt.title('Proportions of colors in the image')
    start = 0
    i = 1
    for p, c in p_and_c:
        end = start + int(p * bar.shape[1])
        if i == clusters:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end
        i += 1

    plt.imshow(bar)
    plt.xticks([])
    plt.yticks([])

    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 250, cols // 2 - 90), (rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)

    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image',
                (rows // 2 - 230, cols // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8, (0, 0, 0), 1, cv2.LINE_AA)

    start = rows // 2 - 220
    for i in range(5):
        end = start + 70
        final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i + 1), (start + 25, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        start = end + 20

    plt.show()
    return p_and_c[1]

import fastai
from fastai.vision.all import *
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from PIL import Image

PATH = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn'
PATH1 = r"C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn"

def load_model():
    path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
    # assert os.path.isfile(path)
    # map_location = torch.device('cpu')
    learn = load_learner(path, cpu=True)
    print(learn)
    return learn

def get_x(r):
    new_path = r["image_name"].replace('\\', '//')
    one_path = os.path.join(PATH1, new_path)
    filename = Path(one_path)
    # print(filename)
    return filename

def get_y(r): return r['labels'].split(',')

def splitter(df):
    train = df.index[df['is_valid'] == 0].tolist()
    valid = df.index[df['is_valid'] == 1].tolist()
    return train, valid

def predict_attribute(model, path, display_img=True):
    predicted = model.predict(path)
    # if display_img:
    #     size = 244,244
    #     img=Image.open(path)
    #     # img.thumbnail(size,Image.ANTIALIAS)
    #     img.show()
    return predicted[0]

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()
class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps: float = 0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#929222
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)

    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"

from fastai.vision.all import DataBlock, ImageBlock, MultiCategoryBlock, RandomResizedCrop, aug_transforms
import pandas as pd
import torch
from fastai.vision.all import *
from functools import partial
def predict_attribute_model(img_path):
    print('alo alo')

    # Define paths to files
    TRAIN_PATH = "multilabel-train.csv"
    TEST_PATH = "multilabel-test.csv"
    CLASSES_PATH = "attribute-classes.txt"

    # Load training data
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        print("Training data loaded successfully.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Parameters and functions for optimization
    wd = 5e-7  # weight decay parameter
    opt_func = partial(ranger, wd=wd)

    # Ensure splitter, get_x, and get_y are defined. For now, I'll assume they are as placeholders.
    splitter = RandomSplitter()  # Example splitter
    get_x = lambda x: x[0]  # Example get_x function
    get_y = lambda x: x[1]  # Example get_y function

    # Define DataBlock
    print('Define datablock')
    try:
        dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                           splitter=splitter,
                           get_x=get_x,
                           get_y=get_y,
                           item_tfms=RandomResizedCrop(224, min_scale=0.8),
                           batch_tfms=aug_transforms())

        dls = dblock.dataloaders(train_df, num_workers=0)
        print('DataBlock and DataLoaders created successfully.')
    except Exception as e:
        print(f"Error creating DataBlock or DataLoaders: {e}")
        return

    # Show a batch of images
    try:
        dls.show_batch(nrows=1, ncols=6)
        print('Batch of images shown.')
    except Exception as e:
        print(f"Error showing batch: {e}")

    # Define metrics
    metrics = [FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]

    # Load test data
    try:
        test_df = pd.read_csv(TEST_PATH)
        print("Test data loaded successfully.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Define a new DataBlock for test data
    try:
        dblock_test = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                                get_x=get_x,
                                get_y=get_y,
                                item_tfms=Resize(224))
        print('Test DataBlock created.')
    except Exception as e:
        print(f"Error creating test DataBlock: {e}")
        return

    # Initialize and load model
    try:
        learn = vision_learner(dls, resnet34, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
                               metrics=metrics, opt_func=opt_func).to_fp16()

        # Load the pre-trained model
        path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
        learn.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model'])
        print('Model loaded successfully.')
    except Exception as e:
        print(f"Error initializing or loading model: {e}")
        return

    # Predict attributes
    try:
        label_result = predict_attribute(learn, img_path)  # Ensure predict_attribute function is defined
        print(f'Prediction result: {label_result}')
        return label_result
    except Exception as e:
        print(f"Error predicting attributes: {e}")
        return






# flask app and routes
@app.route('/')
def home():
    return render_template('welcome.html')

from flaskapp.user.routes import *

@app.route('/login/')
def dologin():
    return render_template('home.html')

@app.route('/register/')
def doregister():
    return render_template('register.html')



@app.route('/outfit/day', methods=['GET', 'POST'])
@login_required
def get_outfit():
    userId = session['user']['_id']
    cityByDefault = 'Bucharest'
    result_outfit = [
        'Dress_Sandal', 'T-shirt/top_Trouser_Sneaker', 'Shirt_Trouser', 'Shirt_Trouser_Sneaker',
        'Dress_Sandal_Coat', 'T-shirt/top_Trouser', 'Shirt_Trouser_Coat', 'Shirt_Trouser_Coat',
        'Dress_Ankle-boot_Coat', 'Pullover_Trouser_Ankle-boot', 'Dress_Sneaker', 'Shirt_Trouser_Sandal',
        'Dress_Sandal_Bag'
    ]

    filter = {'userId': userId}
    if db.city.count_documents(filter) == 0:
        db.city.insert_one({'name': cityByDefault, 'userId': userId})

    cities = db.city.find(filter)
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'

    weather_data = []
    for city in cities:
        r = requests.get(url.format(city['name'])).json()
        weather = {
            'city': city['name'],
            'temperature': r['main']['temp'],
            'description': r['weather'][0]['description'],
            'icon': r['weather'][0]['icon'],
        }
        weather_data.append(weather)

    if len(weather_data) < 3:
        weather_data.extend(
            [{'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''}] * (3 - len(weather_data)))

    city1, city2, city3 = weather_data[:3]

    # Initialize outfits
    outfit1, outfit2, outfit3 = [], [], []

    # Define feature preparation function
    def prepare_features(include_weather, event):
        features = [0] * 10  # Initialize a list with 10 zeros

        # Update features based on weather
        if include_weather == 'yes':
            features[0] = 1  # Assuming the first position is for weather inclusion
            if temperature <= 6.0:
                features[1:6] = [1, 0, 0, 0, 0]
            elif 15.0 < temperature < 26.0:
                features[1:6] = [0, 1, 0, 0, 0]
            elif 6.0 < temperature <= 15.0:
                features[1:6] = [0, 0, 1, 0, 0]
            elif temperature >= 25.0:
                features[1:6] = [0, 0, 0, 1, 0]
        else:
            features[0] = 0  # No weather included
            features[1:6] = [0, 0, 0, 0, 0]

        # Update features based on event
        if event == 'event':
            features[6:10] = [1, 0, 0, 0]
        elif event == 'walk':
            features[6:10] = [0, 1, 0, 0]
        elif event == 'work':
            features[6:10] = [0, 0, 1, 0]
        elif event == 'travel':
            features[6:10] = [0, 0, 0, 1]

        return features

    if request.method == 'POST':
        include_weather = request.form.get('weather')
        city = request.form.get('city')
        event = request.form.get('events')
        option = request.form.get('options')

        if option is not None:
            filter_lookup = {'userId': userId, 'outfitNo': option}
            outfit_rez = db.outfits.find(filter_lookup).sort('_id', -1).limit(1)
            for doc in outfit_rez:
                for piece in doc['outfit']:
                    mydocq = {'_id': piece['_id']}
                    piece['nota'] = piece['nota'] + 1
                    newvalue_doc = {"$set": {"nota": piece['nota']}}
                    db.wardrobe.update_one(mydocq, newvalue_doc)

                doc['nota'] = doc['nota'] + 1
                myquery = {'_id': doc['_id']}
                newvalues = {"$set": {"nota": doc['nota']}}
                newset = {"$set": {"isFavorite": 'yes'}}
                db.outfits.update_one(myquery, newvalues)
                db.outfits.update_one(myquery, newset)

        loaded_classifier = joblib.load("./random_forest.joblib")
        temperature = 20  # Default temperature

        if include_weather == 'yes':
            if city == city1['city']:
                temperature = city1['temperature']
            elif city == city2['city']:
                temperature = city2['temperature']
            elif city == city3['city']:
                temperature = city3['temperature']

        # Prepare the features for prediction
        to_be_predicted = prepare_features(include_weather, event)
        predict_form = [to_be_predicted]

        try:
            # Predict the outfit
            result_forest = loaded_classifier.predict(predict_form)
            index_of_outfit = result_forest[0]
            txt = result_outfit[index_of_outfit]
            filters_outfits = txt.split('_')

            outfit1, outfit2, outfit3 = [], [], []

            for filter_name in filters_outfits:
                filter = {'userId': userId, 'label': filter_name}
                users_clothes = list(db.wardrobe.find(filter))
                if len(users_clothes) >= 3:
                    outfit1.append(users_clothes[0])
                    outfit2.append(users_clothes[1])
                    outfit3.append(users_clothes[2])

            db.outfits.insert_one(
                {'outfit': outfit1, 'userId': userId, 'nota': 4, 'outfitNo': 'piece1', 'isFavorite': 'no'})
            db.outfits.insert_one(
                {'outfit': outfit2, 'userId': userId, 'nota': 4, 'outfitNo': 'piece2', 'isFavorite': 'no'})
            db.outfits.insert_one(
                {'outfit': outfit3, 'userId': userId, 'nota': 4, 'outfitNo': 'piece3', 'isFavorite': 'no'})

        except ValueError as e:
            # Handle the exception if there is an issue with prediction
            print(f"Error during prediction: {e}")
            outfit1, outfit2, outfit3 = [], [], []  # Set empty lists if an error occurs

    return render_template('outfit_of_the_day.html', outfit1=outfit1, outfit2=outfit2, outfit3=outfit3, city1=city1,
                           city2=city2, city3=city3)

@app.route('/dashboard/', methods=['GET', 'POST'])
@login_required
def dashboard():
    userId = session['user']['_id']
    cityByDefault = 'Bucharest'
    api_key = 'aa73cad280fbd125cc7073323a135efa'

    if request.method == 'POST':
        new_city = request.form.get('city')
        print(f"New city submitted: {new_city}")

        if new_city:
            geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={new_city}&limit=1&appid={api_key}'
            geocode_response = requests.get(geocode_url).json()
            print(f"Geocode response: {geocode_response}")

            if geocode_response:
                lat = geocode_response[0].get('lat')
                lon = geocode_response[0].get('lon')
                if lat and lon:
                    db.city.insert_one({'name': new_city, 'lat': lat, 'lon': lon, 'userId': userId})

    filter = {'userId': userId}
    if db.city.find_one(filter) is None:
        geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={cityByDefault}&limit=1&appid={api_key}'
        geocode_response = requests.get(geocode_url).json()
        print(f"Default city geocode response: {geocode_response}")

        if geocode_response:
            lat = geocode_response[0].get('lat')
            lon = geocode_response[0].get('lon')
            if lat and lon:
                db.city.insert_one({'name': cityByDefault, 'lat': lat, 'lon': lon, 'userId': userId})

    cities = db.city.find(filter)
    url = 'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric'

    weather_data = []

    for city in cities:
        if 'lat' not in city or 'lon' not in city:
            geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city["name"]}&limit=1&appid={api_key}'
            geocode_response = requests.get(geocode_url).json()
            print(f"Geocode response for {city['name']}: {geocode_response}")

            if geocode_response:
                lat = geocode_response[0].get('lat')
                lon = geocode_response[0].get('lon')
                if lat and lon:
                    db.city.update_one({'_id': city['_id']}, {'$set': {'lat': lat, 'lon': lon}})
                    city['lat'] = lat
                    city['lon'] = lon
            else:
                continue

        r = requests.get(url.format(city['lat'], city['lon'], api_key)).json()
        print(f"Weather API response for {city['name']}: {r}")

        if r.get('weather') and r.get('main'):
            weather = {
                'city': city['name'],
                'temperature': r['main']['temp'],
                'description': r['weather'][0]['description'],
                'icon': r['weather'][0]['icon'],
            }
            weather_data.append(weather)

    print(f"Weather data to be rendered: {weather_data}")
    return render_template('dashboard.html', weather_data=weather_data)



@app.route('/wardrobe', methods=['GET', 'POST'])
@login_required
def add_wardrobe():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        color = predict_color(file_path)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)

        result = class_names[predicted_label]

    return render_template('wardrobe.html')

@app.route('/wardrobe/all', methods=['GET', 'POST'])
@login_required
def view_wardrobe_all():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId}
    users_clothes = db.wardrobe.find(filter)
    try:
        record = users_clothes.next()
        print(record)
    except StopIteration:
        print("Empty cursor!")

    return render_template('wardrobe_all2.html', wardrobes=users_clothes)


@app.route('/outfits/all', methods=['GET', 'POST'])
@login_required
def view_outfits_all():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId,'isFavorite':'yes'}
    users_clothes = db.outfits.find(filter)
    # for piece in users_clothes:
    #     print(piece)
    #     for ok in piece:
    #         print(ok)
    # for doc in users_clothes:
    #     print("helloooo")
    #     print(doc['outfit'])
    try:
        record = users_clothes.next()
        print(record)
    except StopIteration:
        print("Empty cursor!")

    return render_template('outfits_all.html', wardrobes=users_clothes)

@app.route('/predict', methods=['POST'])
@login_required
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']
            file_path = os.path.join(
                'flaskapp/static/image_users/', secure_filename(f.filename))
            f.save(file_path)
            print(file_path)
            file_path_bd = os.path.join(
                '../flaskapp/static/image_users/', secure_filename(f.filename))

            # Make prediction
            preds = model_predict(file_path, model)
            # aici comentez ultima data
            _, color = predict_color(file_path)
            attribute_predict = predict_attribute_model(file_path)

            mySeparator = ","
            resulted_attribute = "N/A"  # Initialize resulted_attribute
            if attribute_predict is not None:
                resulted_attribute = mySeparator.join(attribute_predict)

            listToStr = ' '.join(map(str, color))
            # listToStr = 'black'  # Assume 'black' for color as a placeholder
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            predicted_label = np.argmax(preds)
            result = class_names[predicted_label]
            userId = session['user']['_id']
            db.wardrobe.insert_one({'label': result, 'attribute': resulted_attribute, 'color': listToStr, 'nota': 4, 'userId': userId,
                                    'file_path': file_path_bd})

            return result
        except Exception as e:
            return str(e), 500
    return None


# calendar logic

from flask import Flask, render_template, request, jsonify, session, redirect, url_for




# ------------------ API: Fetch Wardrobe Items ------------------#
@app.route('/api/wardrobe-items', methods=['GET'])
def get_wardrobe_items():
    """Get wardrobe items grouped by category"""
    user_id = session.get('user', {}).get('_id', '')
    if not user_id:
        return jsonify({"success": False, "message": "User not logged in"}), 400

    items_cursor = db.wardrobe.find({'userId': user_id})

    categories = {
        'tops': ['T-shirt/top', 'Shirt', 'Pullover'],
        'bottoms': ['Trouser'],
        'dresses': ['Dress'],
        'outerwear': ['Coat'],
        'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
        'accessories': ['Bag']
    }

    grouped_items = {cat: [] for cat in categories.keys()}

    for item_doc in items_cursor:
        label = item_doc.get('label', '')
        category = next((cat for cat, lbls in categories.items() if label in lbls), None)
        if category:
            grouped_items[category].append({
                '_id': str(item_doc['_id']),
                'label': label,
                'file_path': normalize_path(item_doc.get('file_path', '')),
                'color': item_doc.get('color', '')
            })

    return jsonify(grouped_items)

# ------------------ API: Fetch Calendar Data ------------------#

# ------------------ API: Add or Edit Outfit ------------------#

# ------------------ API: Delete Outfit ------------------#


@app.route('/debug/outfit/<day>')
@login_required
def debug_outfit(day):
    try:
        user_id = session.get('user', {}).get('_id', '')
        month = request.args.get('month', datetime.now().month)
        year = request.args.get('year', datetime.now().year)

        outfit = db.calendar.find_one({
            'user_id': user_id,
            'day': int(day),
            'month': int(month),
            'year': int(year)
        })

        if outfit:
            # Convert ObjectId to string for JSON serialization
            outfit['_id'] = str(outfit['_id'])

            # Add debugging information
            debug_info = {
                'file_exists': None,
                'full_path': None
            }

            if outfit.get('custom_image'):
                # Get the full path of the image
                full_path = os.path.join(
                    app.static_folder,
                    outfit['custom_image'].lstrip('/static/')
                )
                debug_info['full_path'] = full_path
                debug_info['file_exists'] = os.path.exists(full_path)

            return jsonify({
                'success': True,
                'outfit': outfit,
                'debug_info': debug_info
            })

        return jsonify({
            'success': False,
            'message': 'No outfit found',
            'query': {
                'user_id': user_id,
                'day': int(day),
                'month': int(month),
                'year': int(year)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })






# avatar logic

@app.route('/api/avatar', methods=['GET'])
def get_avatar():
    # Servește modelul din `static/models`
    return send_from_directory('static/models', 'woman.gltf', mimetype='model/gltf+json')
@app.route('/api/update-avatar', methods=['POST'])


@app.route('/avatar')
@login_required
def avatar_page():
    return render_template('avatar.html')


UPLOAD_FOLDER = 'static/image_users/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(os.path.join('flaskapp', UPLOAD_FOLDER), exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


def save_uploaded_image(base64_string, user_id):
    try:
        base64_data = base64_string.split(',')[1] if ',' in base64_string else base64_string
        image_data = base64.b64decode(base64_data)
        images_dir = os.path.join('flaskapp', app.config['UPLOAD_FOLDER'], user_id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'calendar_outfit_{timestamp}.jpg'
        full_path = os.path.join(images_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(image_data)
        print(f"Image saved to: {full_path}")
        return f'/static/image_users/{user_id}/{filename}'
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None


def normalize_path(file_path):
    if not file_path:
        return None
    return file_path.lstrip('/')


@app.route('/calendar/add', methods=['POST'])
@login_required
def add_calendar_outfit():
    try:
        user_id = session.get('user', {}).get('_id', '')
        if not user_id:
            return jsonify({'success': False, 'message': 'User not found in session!'}), 400
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided!'}), 400

        day = int(data.get('day', 0))
        month = int(data.get('month', 0))
        year = int(data.get('year', 0))
        description = data.get('description', '')
        selected_items = data.get('selected_items', [])
        items_object_ids = [ObjectId(i) for i in selected_items if i]

        custom_image_path = None
        uploaded_image = data.get('uploaded_image')
        if uploaded_image:
            custom_image_path = save_uploaded_image(uploaded_image, user_id)

        existing_outfit = db.calendar.find_one({
            'user_id': user_id,
            'day': day,
            'month': month,
            'year': year
        })

        if existing_outfit:
            update_data = {
                'description': description,
                'items': items_object_ids
            }
            if custom_image_path:
                update_data['custom_image'] = custom_image_path
            db.calendar.update_one(
                {'_id': existing_outfit['_id']},
                {'$set': update_data}
            )
            msg = 'Outfit updated successfully!'
        else:
            insert_data = {
                'user_id': user_id,
                'day': day,
                'month': month,
                'year': year,
                'description': description,
                'items': items_object_ids,
                'created_at': datetime.now()
            }
            if custom_image_path:
                insert_data['custom_image'] = custom_image_path
            db.calendar.insert_one(insert_data)
            msg = 'Outfit added successfully!'

        return jsonify({'success': True, 'message': msg})
    except Exception as e:
        print(f"[ERROR] add_calendar_outfit: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/calendar', methods=['GET'])
@login_required
def calendar_view():
    year = int(request.args.get('year', datetime.now().year))
    month = int(request.args.get('month', datetime.now().month))
    user_id = session.get('user', {}).get('_id', '')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 403

    calendar.setfirstweekday(calendar.MONDAY)
    weeks = calendar.monthcalendar(year, month)
    outfit_map = {}

    try:
        outfits_cursor = db.calendar.find({"user_id": user_id, "year": year, "month": month})
        for outfit_doc in outfits_cursor:
            day_number = outfit_doc['day']
            outfit_items = []
            if 'items' in outfit_doc and isinstance(outfit_doc['items'], list):
                for item_id in outfit_doc['items']:
                    item_obj = db.wardrobe.find_one({'_id': ObjectId(item_id)})
                    if item_obj:
                        outfit_items.append({
                            'id': str(item_obj['_id']),
                            'label': item_obj['label'],
                            'file_path': normalize_path(item_obj.get('file_path', '')),
                            'color': item_obj.get('color', '')
                        })
            custom_image = normalize_path(outfit_doc.get('custom_image'))
            print(f"Normalized custom image path: {custom_image}")
            if custom_image:
                full_path = os.path.join('flaskapp', custom_image.lstrip('/'))
                print(f"Full path: {full_path}")
                print(f"File exists: {os.path.exists(full_path)}")
            outfit_map[day_number] = {
                'id': str(outfit_doc['_id']),
                'outfit_items': outfit_items,
                'description': outfit_doc.get('description', ''),
                'custom_image': custom_image
            }
    except Exception as e:
        print(f"Error loading outfits: {str(e)}")
        outfit_map = {}

    return render_template(
        'calendar.html',
        year=year,
        month=month,
        weeks=weeks,
        outfits=outfit_map,
        month_names=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
    )


@app.route('/calendar/delete', methods=['DELETE'])
def delete_calendar_outfit():
    """Delete an outfit from the calendar."""
    user_id = session.get('user', {}).get('_id', '')
    day, month, year = int(request.args.get('day')), int(request.args.get('month')), int(request.args.get('year'))

    result = db.calendar.delete_one({"user_id": user_id, "day": day, "month": month, "year": year})

    if result.deleted_count > 0:
        return jsonify({"success": True, "message": "Outfit deleted successfully!"})
    return jsonify({"success": False, "message": "Outfit not found!"}), 404






if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
