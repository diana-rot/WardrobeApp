from __future__ import division, print_function
import os
import gc
import numpy as np
from flask import render_template, request, session, url_for
from flaskapp import app, login_required,redirect
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import pymongo
import requests
from gridfs import GridFS
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from keras.preprocessing import image


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import joblib

client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test

import cv2

from sklearn.cluster import KMeans
import imutils

MODEL_PATH = 'my_second_model.h5'
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

# def predict_attribute_model(img_path):
#     print('alo alo')
#     TRAIN_PATH = "multilabel-train.csv"
#     TEST_PATH = "multilabel-test.csv"
#     CLASSES_PATH = "attribute-classes.txt"
#
#     train_df = pd.read_csv(TRAIN_PATH)
#
#     train_df.head()
#     wd = 5e-7  # weight decay parameter
#     opt_func = partial(ranger, wd=wd)
#     print('pana aci ok+2')
#
#     # aci e buba
#     dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
#                        splitter=splitter,
#                        get_x=get_x,
#                        get_y=get_y,
#                        item_tfms=RandomResizedCrop(224, min_scale=0.8),
#                        batch_tfms=aug_transforms())
#
#
#     dls = dblock.dataloaders(train_df, num_workers=0)
#     print('pana aci ok--3')
#     dls.show_batch(nrows=1, ncols=6)
#     print('pana aci ok+3')
#
#     # dsets = dblock.datasets(train_df)
#     metrics = [FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]
#     print('pana aci ok+4')
#     test_df = pd.read_csv(TEST_PATH)
#     test_df.head()
#     print('pana aci ok+5')
#     dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
#                        get_x=get_x,
#                        get_y=get_y,
#                        item_tfms=Resize(224))
#     print('pana aci ok+6')
#
#     print(dls.vocab)
#     print('pana aci ok+7')
#     learn = vision_learner(dls, resnet34, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
#                            metrics=metrics, opt_func=opt_func).to_fp16()
#
#     path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
#
#     learn.load_state_dict(torch.load(path,
#                                      map_location=torch.device('cpu'))['model'])
#     label_result = predict_attribute(learn, img_path)
#     print(label_result)
#     return label_result


@app.route('/')
def home():
    return render_template('welcome.html')


from flaskapp.user.routes import *


@app.route('/login/')
def dologin():
    return render_template('home.html')


from flaskapp.user.routes import *


@app.route('/register/')
def doregister():
    return render_template('register.html')



@app.route('/outfit/day', methods=['GET', 'POST'])
@login_required
def get_outfit():
    userId = session['user']['_id']
    cityByDefault = 'Bucharest'
    result_outfit = []
    result_outfit.append('Dress_Sandal')#0
    result_outfit.append('T-shirt/top_Trouser_Sneaker')#1
    result_outfit.append('Shirt_Trouser')#2
    result_outfit.append('Shirt_Trouser_Sneaker')# 3
    result_outfit.append('Dress_Sandal_Coat')# 4
    result_outfit.append('T-shirt/top_Trouser')# 5
    result_outfit.append('Shirt_Trouser_Coat')# 6
    result_outfit.append('Shirt_Trouser_Coat')# 7
    result_outfit.append('Dress_Ankle-boot_Coat')  # 8
    result_outfit.append('Pullover_Trouser_Ankle-boot')# 9
    result_outfit.append('Dress_Sneaker')  # 10
    result_outfit.append('Shirt_Trouser_Sandal')# 11
    result_outfit.append('Dress_Sandal_Bag') #12


    filter = {'userId': userId}
    if db.city.find(filter) is None:
        db.city.insert_one({'name': cityByDefault, 'userId': userId})

    cities = db.city.find(filter)
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'


    weather_data = []
    outfit1 = []
    outfit2 = []
    outfit3 = []


    for city in cities:
        r = requests.get(url.format(city['name'])).json()

        weather = {
            'city': city['name'],
            'temperature': r['main']['temp'],
            'description': r['weather'][0]['description'],
            'icon': r['weather'][0]['icon'],
        }
        weather_data.append(weather)


    city1 = weather_data[0]
    city2 = weather_data[1]
    city3 = weather_data[2]

    if request.method == 'POST':
        include_weather = request.form.get('weather')
        print(include_weather)
        city = request.form.get('city')
        print(city)
        event = request.form.get('events')
        print(event)
        option = request.form.get('options')
        print(option)
        #take the last introduced outfit and modify the score

        if option is not None:
            filter_lookup = {'userId': userId, 'outfitNo': option}
            outfit_rez= db.outfits.find(filter_lookup).sort('_id', -1).limit(1);
            #outfit_rez = db.outfits.find(filter_lookup).limit(1)
            print(filter)
            print(filter_lookup)
            print('hello')
            print(outfit_rez)
            #for every piece of clothing in the outfit, update the score of the clothing
            for doc in outfit_rez:
                print(doc)
                print(doc['nota'])
                print(doc['outfit'])
                for piece in doc['outfit']:
                    print(piece['label'])
                    print(piece['_id'])
                    mydocq = {'_id': piece['_id']}
                    piece['nota'] = piece['nota'] + 1
                    newvalue_doc = {"$set": {"nota": piece['nota']}}
                    db.wardrobe.update_one(mydocq, newvalue_doc)


                doc['nota'] = doc['nota'] + 1
                print(doc['nota'])
                myquery = {'_id': doc['_id']}
                print(myquery)
                newvalues = {"$set": {"nota": doc['nota']}}
                newset = {"$set": {"isFavorite": 'yes'}}
                db.outfits.update_one(myquery, newvalues)
                db.outfits.update_one(myquery, newset)


        #Random forect classificer
        loaded_classifier = joblib.load("./random_forest_classifier.joblib")
        #to be predicted - users preferences
        to_be_predicted = []

        if include_weather == 'yes':
            to_be_predicted.append(1)

            if city == city1['city']:
                temperature = city1['temperature']
            elif city == city2['city']:
                temperature = city2['temperature']
            elif city == city3['city']:
                temperature = city3['temperature']

#includ si raining dupa ce fac un rezultat aproape de perfect

            if temperature <= 6.0:
                print('iarna')
                to_be_predicted.append(1)
                to_be_predicted.append(0)
                to_be_predicted.append(0)
                to_be_predicted.append(0)
                to_be_predicted.append(0)

            elif temperature > 15.0 and temperature < 26.0:
                print('primavara')
                to_be_predicted.append(0)
                to_be_predicted.append(1)
                to_be_predicted.append(0)
                to_be_predicted.append(0)
                to_be_predicted.append(0)

            elif temperature > 6.0 and temperature <= 15.0:
                 print('toamna')
                 to_be_predicted.append(0)
                 to_be_predicted.append(0)
                 to_be_predicted.append(1)
                 to_be_predicted.append(0)
                 to_be_predicted.append(0)

            elif temperature >= 25.0:
                print('vara')
                to_be_predicted.append(0)
                to_be_predicted.append(0)
                to_be_predicted.append(0)
                to_be_predicted.append(1)
                to_be_predicted.append(0)


        elif include_weather == 'no':
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)

        if event == 'event':
            to_be_predicted.append(1)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
        elif event == 'walk':
            to_be_predicted.append(0)
            to_be_predicted.append(1)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
        elif event == 'work':
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(1)
            to_be_predicted.append(0)
        elif event == 'travel':
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(0)
            to_be_predicted.append(1)

        print(to_be_predicted)
        predict_form = []
        #aici il formatez sa il trimit la padurea de arbori
        predict_form.append(to_be_predicted)
        #result forest are indexul sub forma de vector
        if event is not None:

            print(predict_form)
            result_forest = loaded_classifier.predict(predict_form)
            print(result_forest)
            index_of_outfit = result_forest[0]
            print(result_outfit[index_of_outfit])

            #the results to be separated for the FE
            txt = result_outfit[index_of_outfit]
            filters_outfits = txt.split('_')
            print(filters_outfits)

            #if we already have some favourites
            for filter_name in filters_outfits:
                print(filter_name)

                #nu merge, trebuie sa gasesc o alta cale
                #facem asa, scot nota momentan de aici, ma duc sa printez documentul si pana m
                #ma intorc, am si timp sa ma gandesc
                # {"$lt": 5}
                filter = {'userId': userId, 'label': filter_name}
                print("here it suppose to go")
                print(filter)
                count = 0
                #each item of clothing
                users_clothes = db.wardrobe.find(filter).limit(1)
                print(users_clothes)
                outfit1.append(users_clothes[1])
                outfit2.append(users_clothes[2])
                outfit3.append(users_clothes[3])
                    # print(count)
                print('are you ok?1')
                print(outfit1)

                option1 = outfit1
                print('are you ok?2')
                print(outfit2)

                option2 = outfit1
                print('are you ok?3')
                print(outfit3)

                option3 = outfit3

            db.outfits.insert_one(
                    {'outfit': outfit1, 'userId': userId, 'nota': 4, 'outfitNo':'piece1', 'isFavorite':'no'})
            db.outfits.insert_one(
                    {'outfit': outfit2, 'userId': userId, 'nota': 4, 'outfitNo':'piece2','isFavorite':'no'})
            db.outfits.insert_one(
                    {'outfit': outfit3, 'userId': userId, 'nota': 4, 'outfitNo':'piece3','isFavorite':'no'})


    return render_template('outfit_of_the_day.html',  outfit1 = outfit1, outfit2= outfit2, outfit3= outfit3, city1=city1, city2=city2, city3=city3)



@app.route('/dashboard/', methods=['GET', 'POST'])
@login_required
def dashboard():
    userId = session['user']['_id']
    cityByDefault = 'Bucharest'

    if request.method == 'POST':
        new_city = request.form.get('city')

        if new_city:
            db.city.insert_one({'name': new_city, 'userId': userId})

    filter = {'userId': userId}
    if db.city.find(filter) is None:
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

    return render_template('dashboard.html', weather_data=weather_data)


from flaskapp.user.routes import *


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

    return render_template('wardrobe_all.html', wardrobes=users_clothes)


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
                '../static/image_users/', secure_filename(f.filename))

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



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
