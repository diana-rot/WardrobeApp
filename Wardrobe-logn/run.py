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


from flask import Flask, render_template, request, jsonify, session, redirect, url_for




client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test
DEFAULT_RATING = 4

import cv2
from sklearn.cluster import KMeans
import imutils

# MODEL_PATH = 'my_second_model.h5'
MODEL_PATH = 'improved_fashion_model.h5'
# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(28, 28))
        img_array = np.asarray(img)
        x = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        result = int(img_array[0][0][0])

        img = cv2.bitwise_not(x) if result > 128 else x
        img = img / 255
        img = np.expand_dims(img, 0)

        preds = model.predict(img)
        if preds is None or len(preds) == 0:
            raise ValueError("Model prediction returned None or empty")

        return preds

    except Exception as e:
        print(f"Error in model_predict: {str(e)}")
        raise

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


@app.route('/predict', methods=['POST'])
@login_required
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            if not f:
                return "No file uploaded", 400

            user_id = session['user']['_id']
            upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Make predictions with validation
            try:
                preds = model_predict(file_path, model)
                if not isinstance(preds, np.ndarray) or preds.size == 0:
                    raise ValueError("Invalid prediction output")

                color_result = predict_color(file_path)
                if not color_result or len(color_result) < 2:
                    raise ValueError("Invalid color prediction")

                predicted_label = np.argmax(preds)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                if predicted_label >= len(class_names):
                    raise ValueError("Invalid predicted label index")

                result = class_names[predicted_label]

                # Save to database
                db.wardrobe.insert_one({
                    'label': result,
                    'color': ' '.join(map(str, color_result[1])),
                    'nota': 4,
                    'userId': user_id,
                    'file_path': f'/static/image_users/{user_id}/{secure_filename(f.filename)}'
                })

                return result

            except Exception as e:
                print(f"Prediction error: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return str(e), 500

        except Exception as e:
            return str(e), 500

    return None
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

# @app.route('/user/signOut')
# def signout():
#     # Clear the session
#     session.clear()
#     # Redirect to the login page
#     return redirect(url_for('dologin'))
#profile
# Add these imports to your existing imports
import base64
from bson.binary import Binary


# Add this route to handle profile updates
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            user_id = session['user']['_id']

            # Handle profile picture upload
            if 'profile_picture' in request.files:
                file = request.files['profile_picture']
                if file and allowed_file(file.filename):
                    # Read the file and convert to binary for MongoDB storage
                    file_data = file.read()

                    # Create upload directory if it doesn't exist
                    upload_dir = os.path.join('flaskapp', 'static', 'profile_pictures', str(user_id))
                    os.makedirs(upload_dir, exist_ok=True)

                    # Save file to disk
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(upload_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(file_data)

                    # Update user document with profile picture path
                    db.users.update_one(
                        {'_id': user_id},
                        {'$set': {
                            'profile_picture': f'/static/profile_pictures/{user_id}/{filename}'
                        }}
                    )

            # Update other profile information
            name = request.form.get('name')
            email = request.form.get('email')

            # Update user document
            update_data = {}
            if name:
                update_data['name'] = name
            if email:
                update_data['email'] = email

            if update_data:
                db.users.update_one(
                    {'_id': user_id},
                    {'$set': update_data}
                )

                # Update session data
                user = db.users.find_one({'_id': user_id})
                session['user'] = user

            return jsonify({'success': True, 'message': 'Profile updated successfully'})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    # GET request - display profile page
    user_data = db.users.find_one({'_id': session['user']['_id']})
    return render_template('profile.html', user=user_data)



# Helper function to get user's profile picture
def get_user_profile_picture():
    if 'user' in session and session['user']:
        user = db.users.find_one({'_id': session['user']['_id']})
        return user.get('profile_picture', '/static/image/default-profile.png')
    return '/static/image/default-profile.png'


# Add this to your template context
@app.context_processor
def utility_processor():
    return dict(get_user_profile_picture=get_user_profile_picture)

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






@app.route('/wardrobe/delete/<item_id>', methods=['DELETE'])
@login_required
def delete_wardrobe_item(item_id):
    try:
        userId = session['user']['_id']
        # Get item to delete its image file
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': userId})

        if not item:
            return jsonify({'error': 'Item not found'}), 404

        # Delete physical file if it exists
        file_path = os.path.join('flaskapp', item['file_path'].lstrip('/'))
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete from database
        result = db.wardrobe.delete_one({'_id': ObjectId(item_id), 'userId': userId})

        if result.deleted_count:
            return jsonify({'success': True})
        return jsonify({'error': 'Delete failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# First, define the helper function
DEFAULT_RATING = 4


def prepare_features(include_weather, event, temperature):
    """Prepare features for model prediction"""
    features = [0] * 10

    # Weather features
    if include_weather == 'yes':
        features[0] = 1
        if temperature <= 6.0:
            features[1:6] = [1, 0, 0, 0, 0]
        elif 6.0 < temperature <= 15.0:
            features[1:6] = [0, 1, 0, 0, 0]
        elif 15.0 < temperature < 26.0:
            features[1:6] = [0, 0, 1, 0, 0]
        elif temperature >= 26.0:
            features[1:6] = [0, 0, 0, 1, 0]
    else:
        features[0] = 0
        features[1:6] = [0, 0, 0, 0, 0]

    # Event features
    event_mapping = {
        'event': [1, 0, 0, 0],
        'walk': [0, 1, 0, 0],
        'work': [0, 0, 1, 0],
        'travel': [0, 0, 0, 1]
    }
    features[6:10] = event_mapping.get(event, [0, 0, 0, 0])

    return features


@app.route('/outfit/day', methods=['GET', 'POST'])
@login_required
def get_outfit():
    print("Debug: Entering get_outfit route")

    try:
        userId = session['user']['_id']
        print(f"Debug: User ID: {userId}")

        cityByDefault = 'Bucharest'
        DEFAULT_RATING = 4

        # Default to show generator and hide outfits
        show_generator = True
        show_outfits = False
        success_message = None
        error_message = None

        # Define available outfit combinations
        result_outfit = [
            'Dress_Sandal', 'T-shirt/top_Trouser_Sneaker', 'Shirt_Trouser',
            'Shirt_Trouser_Sneaker', 'Dress_Sandal_Coat', 'T-shirt/top_Trouser',
            'Shirt_Trouser_Coat', 'Shirt_Trouser_Coat', 'Dress_Ankle-boot_Coat',
            'Pullover_Trouser_Ankle-boot', 'Dress_Sneaker', 'Shirt_Trouser_Sandal',
            'Dress_Sandal_Bag'
        ]

        # Initialize city if not exists
        filter = {'userId': userId}
        if db.city.count_documents(filter) == 0:
            print(f"Debug: Creating new city entry for user {userId}")
            db.city.insert_one({'name': cityByDefault, 'userId': userId})

        # Get weather data
        cities = db.city.find(filter)
        weather_data = []
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'

        for city in cities:
            try:
                print(f"Debug: Fetching weather for {city['name']}")
                r = requests.get(url.format(city['name']), timeout=5).json()
                weather = {
                    'city': city['name'],
                    'temperature': r['main']['temp'],
                    'description': r['weather'][0]['description'],
                    'icon': r['weather'][0]['icon'],
                }
                weather_data.append(weather)
            except Exception as e:
                print(f"Error fetching weather for {city['name']}: {e}")
                weather_data.append({
                    'city': city['name'],
                    'temperature': 20,
                    'description': '',
                    'icon': ''
                })

        # Ensure we have 3 weather options
        while len(weather_data) < 3:
            weather_data.append({
                'city': cityByDefault,
                'temperature': 20,
                'description': '',
                'icon': ''
            })

        city1, city2, city3 = weather_data[:3]
        outfit1, outfit2, outfit3 = [], [], []

        if request.method == 'POST':
            print("Debug: Processing POST request")

            # Handle outfit selection
            option = request.form.get('options')
            if option:
                print(f"Debug: Selected option: {option}")
                filter_lookup = {'userId': userId, 'outfitNo': option}
                outfit_doc = db.outfits.find_one(filter_lookup, sort=[('_id', -1)])

                if outfit_doc:
                    # Update outfit pieces ratings
                    for piece in outfit_doc['outfit']:
                        try:
                            current_piece = db.wardrobe.find_one({'_id': piece['_id']})
                            current_rating = current_piece.get('nota',
                                                               DEFAULT_RATING) if current_piece else DEFAULT_RATING

                            db.wardrobe.update_one(
                                {'_id': piece['_id']},
                                {'$set': {'nota': current_rating + 1}},
                                upsert=True
                            )
                        except Exception as e:
                            print(f"Error updating piece rating: {str(e)}")

                    try:
                        # Update outfit rating
                        current_outfit_rating = outfit_doc.get('nota', DEFAULT_RATING)
                        db.outfits.update_one(
                            {'_id': outfit_doc['_id']},
                            {
                                '$set': {
                                    'nota': current_outfit_rating + 1,
                                    'isFavorite': 'yes'
                                }
                            }
                        )
                        # Show success message and hide outfits
                        success_message = "Outfit has been saved to your favorites!"
                        show_outfits = False
                        return render_template(
                            'outfit_of_the_day.html',
                            success_message=success_message,
                            show_generator=show_generator,
                            show_outfits=show_outfits,
                            city1=city1,
                            city2=city2,
                            city3=city3
                        )
                    except Exception as e:
                        print(f"Error updating outfit rating: {str(e)}")
                        error_message = "Error saving outfit. Please try again."

            # Generate new outfits
            include_weather = request.form.get('weather') == 'yes'
            city = request.form.get('city')
            event = request.form.get('events')
            temperature = 20  # Default temperature

            print(f"Debug: Form data - weather: {include_weather}, city: {city}, event: {event}")

            if include_weather and city:
                selected_weather = next(
                    (w for w in weather_data if w['city'] == city),
                    {'temperature': 20}
                )
                temperature = selected_weather['temperature']

            try:
                loaded_classifier = joblib.load("./random_forest.joblib")
                features = prepare_features(include_weather, event, temperature)
                result_forest = loaded_classifier.predict([features])
                index_of_outfit = result_forest[0]
                outfit_combination = result_outfit[index_of_outfit]
                filters_outfits = outfit_combination.split('_')

                print(f"Debug: Generated outfit combination: {outfit_combination}")

                # Generate three outfits
                for i, outfit_list in enumerate([outfit1, outfit2, outfit3]):
                    outfit_pieces = []
                    for filter_name in filters_outfits:
                        clothes = list(db.wardrobe.find({
                            'userId': userId,
                            'label': filter_name
                        }).sort('nota', -1))

                        if clothes:
                            index = min(i, len(clothes) - 1)
                            piece = clothes[index]
                            if not piece.get('file_path'):
                                piece['file_path'] = None
                            if 'nota' not in piece:
                                piece['nota'] = DEFAULT_RATING
                                db.wardrobe.update_one(
                                    {'_id': piece['_id']},
                                    {'$set': {'nota': DEFAULT_RATING}}
                                )
                            outfit_pieces.append(piece)

                    if outfit_pieces:
                        outfit_doc = {
                            'outfit': outfit_pieces,
                            'userId': userId,
                            'nota': DEFAULT_RATING,
                            'outfitNo': f'piece{i + 1}',
                            'isFavorite': 'no',
                            'created_at': datetime.now()
                        }
                        db.outfits.insert_one(outfit_doc)

                        if i == 0:
                            outfit1 = outfit_pieces
                        elif i == 1:
                            outfit2 = outfit_pieces
                        else:
                            outfit3 = outfit_pieces

                show_outfits = True

            except Exception as e:
                print(f"Error generating outfits: {e}")
                error_message = "Error generating outfits. Please try again."

        print("Debug: Rendering template")
        return render_template(
            'outfit_of_the_day.html',
            outfit1=outfit1,
            outfit2=outfit2,
            outfit3=outfit3,
            city1=city1,
            city2=city2,
            city3=city3,
            show_generator=show_generator,
            show_outfits=show_outfits,
            success_message=success_message,
            error_message=error_message
        )

    except Exception as e:
        print(f"Error in get_outfit: {str(e)}")
        return render_template(
            'outfit_of_the_day.html',
            error_message="An error occurred. Please try again.",
            show_generator=True,
            show_outfits=False,
            city1={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
            city2={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
            city3={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''}
        )


@app.route('/wardrobe', methods=['GET', 'POST'])
@login_required
def add_wardrobe():
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            f = request.files['file']
            if f.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Check if the file is allowed
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if not '.' in f.filename or \
                    f.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({'error': 'Invalid file type'}), 400

            # Save the file
            user_id = session['user']['_id']
            upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
            print('dir' + upload_dir);
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)
            print(file_path + 'fsss')
            file_path_db = f'/static/image_users/{user_id}/{secure_filename(f.filename)}'
            print('fileeeDB'+ file_path_db);
            # Rest of the code remains the same...

            try:
                # Make prediction
                preds = model_predict(file_path, model)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                # Get predicted label and confidence scores
                predicted_label = np.argmax(preds)
                clothing_type = class_names[predicted_label]

                # Get color prediction
                color_result = predict_color(file_path)
                # color_result is a tuple of (percentage, RGB array)
                color_percentage = float(color_result[0])
                color_rgb = color_result[1].tolist()  # Convert numpy array to list

                # Save image data
                userId = session['user']['_id']
                db.wardrobe.insert_one({
                    'userId': userId,
                    'label': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    },
                    'filename': secure_filename(f.filename),
                    'file_path': file_path_db,
                    'created_at': datetime.now(),
                    'last_worn': None,
                    'times_worn': 0
                })

                # # Clean up the uploaded file
                # os.remove(file_path)

                # Return success response
                return jsonify({
                    'success': True,
                    'prediction': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    }
                })

            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        except Exception as e:
            print(f"Error in file upload: {str(e)}")
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

    # GET request
    return render_template('wardrobe.html')

@app.route('/wardrobe/all', methods=['GET', 'POST'])
@login_required
def view_wardrobe_all():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId}
    users_clothes = db.wardrobe.find(filter)
    return render_template('wardrobe_all2.html', wardrobes=users_clothes)


@app.route('/outfits/all', methods=['GET', 'POST'])
@login_required
def view_outfits_all():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId, 'isFavorite': 'yes'}
    users_clothes = db.outfits.find(filter)
    return render_template('outfits_all.html', wardrobes=users_clothes)







# avatar logic
@app.route('/api/avatar/<gender>', methods=['GET'])
@login_required
def get_avatar(gender):
    if gender not in ['male', 'female']:
        return jsonify({'error': 'Invalid gender specified'}), 400

    model_file = f'{gender}.gltf'
    return send_from_directory(app.config['MODELS_FOLDER'], model_file, mimetype='model/gltf+json')


@app.route('/api/clothes', methods=['GET'])
@login_required
def get_available_clothes():
    # Return list of available clothing items for the user
    clothes_path = os.path.join(app.config['MODELS_FOLDER'], 'clothing')
    available_clothes = []

    for filename in os.listdir(clothes_path):
        if allowed_file(filename):
            clothes_id = filename.rsplit('.', 1)[0]
            available_clothes.append({
                'id': clothes_id,
                'name': clothes_id.replace('_', ' ').title(),
                'path': f'/api/models/clothing/{filename}'
            })

    return jsonify(available_clothes)



def get_user_preferences(user_id):
    try:
        # You can replace this with your database query
        with open(f'user_preferences/{user_id}.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Return default preferences if none exist
        return {
            'gender': 'female',
            'skinColor': '#FFE0BD',
            'hairColor': '#000000',
            'hairStyle': 'default',
            'clothes': []
        }


def save_user_preferences(user_id, preferences):
    # You can replace this with your database save logic
    with open(f'user_preferences/{user_id}.json', 'w') as file:
        json.dump(preferences, file)


@app.route('/avatar')
def avatar_page():
    user_id = session.get('user_id')
    user_preferences = get_user_preferences(user_id)
    return render_template('avatar.html', user_preferences=user_preferences)


@app.route('/api/avatar/customize', methods=['POST'])
def customize_avatar():
    user_id = session.get('user_id')
    data = request.get_json()

    # Update user preferences
    user_preferences = {
        'gender': data.get('gender', 'female'),
        'skinColor': data.get('skinColor', '#FFE0BD'),
        'hairColor': data.get('hairColor', '#000000'),
        'hairStyle': data.get('hairStyle', 'default'),
        'clothes': data.get('clothes', [])
    }

    # Save the updated preferences
    save_user_preferences(user_id, user_preferences)

    return jsonify({'status': 'success', 'preferences': user_preferences})








# calendar logic

UPLOAD_FOLDER = 'static/image_users/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(os.path.join('flaskapp', UPLOAD_FOLDER), exist_ok=True)

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
    return file_path.replace('/static/static/', '/static/').lstrip('/')

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
