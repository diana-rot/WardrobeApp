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

from keras.models import load_model
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

# Image loader
# Image loader
import cv2

from sklearn.cluster import KMeans
import imutils

MODEL_PATH = 'my_1_july_20epc_model.h5'
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

    print(preds)

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
    # print(p_and_c[1][0])
    print(p_and_c[1])

    # print("one col" +p_and_c[0])
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


def predict_attribute_model(img_path):
    print('alo alo')
    TRAIN_PATH = "multilabel-train.csv"
    TEST_PATH = "multilabel-test.csv"
    CLASSES_PATH = "attribute-classes.txt"

    train_df = pd.read_csv(TRAIN_PATH)
    train_df.head()
    wd = 5e-7  # weight decay parameter
    opt_func = partial(ranger, wd=wd)

    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       splitter=splitter,
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms=RandomResizedCrop(224, min_scale=0.8),
                       batch_tfms=aug_transforms())

    dls = dblock.dataloaders(train_df, num_workers=0)
    dls.show_batch(nrows=1, ncols=6)

    # dsets = dblock.datasets(train_df)
    metrics = [FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]

    test_df = pd.read_csv(TEST_PATH)
    test_df.head()
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms=Resize(224))  # Not Sure)

    # test_dls = dblock.dataloaders(test_df, num_workers=0)

    print(dls.vocab)
    learn = vision_learner(dls, resnet34, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
                           metrics=metrics, opt_func=opt_func).to_fp16()

    path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'

    learn.load_state_dict(torch.load(path,
                                     map_location=torch.device('cpu'))['model'])
    label_result = predict_attribute(learn, img_path)
    print(label_result)
    return label_result


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

#
# @app.route('/generate/outfit', methods=['GET', 'POST'])
# @login_required
# def generate_outfit():
#
#     userId = session['user']['_id']
#     print(userId)
#     filter = {'userId': userId, 'label': 'Dress'}
#     users_some_clothes = db.wardrobe.find(filter).limit(3)
#     print("clothes, buth how many?")
#     count = 0;
#
#     clothes1 = users_some_clothes[1]
#     clothes2 = users_some_clothes[2]
#     clothes3 = users_some_clothes[3]
#     print(clothes1)
#     print(clothes2)
#     print(clothes3)
#
#     option = request.form.getlist('options')
#     print(option)
#
#     userId = session['user']['_id']
#     cityByDefault = 'Bucharest'
#
#     if request.method == 'POST':
#         new_city = request.form.get('city')
#
#         if new_city:
#             db.city.insert_one({'name': new_city, 'userId': userId})
#
#     filter = {'userId': userId}
#     if db.city.find(filter) is None:
#         db.city.insert_one({'name': cityByDefault, 'userId': userId})
#
#     cities = db.city.find(filter)
#     url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'
#
#     weather_data = []
#
#     for city in cities:
#         r = requests.get(url.format(city['name'])).json()
#
#         weather = {
#             'city': city['name'],
#             'temperature': r['main']['temp'],
#             'description': r['weather'][0]['description'],
#             'icon': r['weather'][0]['icon'],
#         }
#
#         weather_data.append(weather)
#     city1 = weather_data[0]
#     city2 = weather_data[1]
#     city3 = weather_data[2]
#
#     result_weather = request.form.getlist('city')
#     result_location = request.form.getlist('location')
#     print(result_weather)
#     print(result_location)
#
#
#     # aici e random classifier
#
#     loaded_classifier = joblib.load("./random_forest.joblib")
#     load_clas1 = loaded_classifier.predict([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
#     load_clas2 = loaded_classifier.predict([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0]])
#     print(load_clas1, load_clas2)
#
#     return render_template('outfit_generator.html', outfit1=clothes1, outfit2=clothes2, outfit3=clothes3,
#                            city1=city1, city2=city2, city3=city3)


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
    result_outfit.append('Pullover_Trouser_Ankle-boot')  # 9
    result_outfit.append('Dress_Sneaker')  # 10
    result_outfit.append('Shirt_Trouser_Sandal')# 11
    result_outfit.append('Dress_Sandal_Bag') #12



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
        if option is not None:
            filter_lookup = {'userId': userId, 'outfitNo': option}
            # print(filter_lookup)
            # outfit_rez= db.outfits.getLastInsertedDocument.find(filter_lookup).limit(1);
            outfit_rez = db.outfits.find(filter_lookup).limit(1)
            print(filter)
            print(filter_lookup)
            for doc in outfit_rez:
                print(doc)
                print(doc['nota'])
                print(doc['outfit'])
                for piece in doc['outfit']:
                    print(piece['label'])
                    print(piece['_id'])


                doc['nota'] = doc['nota'] + 1
                print(doc['nota'])
                myquery = {'_id': doc['_id']}
                print(myquery)
                newvalues = {"$set": {"nota": doc['nota']}}

                db.outfits.update_one(myquery, newvalues)



        loaded_classifier = joblib.load("./random_forest_classifier.joblib")
        #to be predicted sunt preferintele utilizatorului
        to_be_predicted = []

        if include_weather == 'yes':
            to_be_predicted.append(1)
                #aici trebuie sa adaug vremea corespunzatoare orasului cu WeatehrOpenAPI
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
            print("not none?")
            print(predict_form)
            result_forest = loaded_classifier.predict(predict_form)
            print(result_forest)
            index_of_outfit = result_forest[0]
            print(result_outfit[index_of_outfit])
        #de aici va trebui sa fac un strtok si sa iau cele 3 outfituri pentru a le afisa in front end
        #gandit logica cu

            txt = result_outfit[index_of_outfit]
            filters_outfits = txt.split('_')
            print(filters_outfits)

            for filter_name in filters_outfits:
                print(filter_name)
                filter = {'userId': userId, 'label': filter_name}
                print(filter)
                count = 0
                # while count != 3:
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
                    {'outfit': outfit1, 'userId': userId, 'nota': 5, 'outfitNo':'piece1'})
            db.outfits.insert_one(
                    {'outfit': outfit2, 'userId': userId, 'nota': 5, 'outfitNo':'piece2'})
            db.outfits.insert_one(
                    {'outfit': outfit3, 'userId': userId, 'nota': 5, 'outfitNo':'piece3'})


    return render_template('outfit_of_the_day.html',  outfit1 = outfit1, outfit2= outfit2, outfit3= outfit3, city1=city1, city2=city2, city3=city3)


@app.route("/outfit/day", methods=['POST'])
def post_result(outfit1,outfit2,outfit3,city1,city2,city3):

    print('help me')
    print(outfit1)
    print(outfit2)
    return render_template('outfit_of_the_day.html',outfit1 = outfit1, outfit2= outfit2, outfit3= outfit3, city1=city1, city2=city2, city3=city3)


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
        # color = predict_color(file_path)
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


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join(
            'flaskapp/static/image_users/', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        file_path_bd = os.path.join(
            '../static/image_users/', secure_filename(f.filename))

        # Make predictionoutfit
        preds = model_predict(file_path, model)
        _, color = predict_color(file_path)
        attribute_predict = predict_attribute_model(file_path)
        print(attribute_predict)
        mySeparator = ","
        if attribute_predict is not None:
            resulted_attribute = mySeparator.join(attribute_predict)

        listToStr = ' '.join(map(str, color))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        userId = session['user']['_id']
        db.wardrobe.insert_one({'label': result, 'attribute': resulted_attribute, 'color': listToStr,'nota':4, 'userId': userId,
                                'file_path': file_path_bd})

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
