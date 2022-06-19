from __future__ import division, print_function
import os
import gc
import numpy as np
from flask import render_template, request, session
from flaskapp import app, login_required
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import pymongo
import requests
from gridfs import GridFS

from keras.models import load_model
from keras.preprocessing import image

client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test

# Image loader
# Image loader
import cv2

from sklearn.cluster import KMeans
import imutils

# Model saved with Keras model.save()
MODEL_PATH = 'my_second_model.h5'
# MODEL_PATH = 'my_model.h5'
#MODEL_PATH ='my_2_class_model.h5'
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

@app.route('/generate/outfit', methods=['GET', 'POST'])
@login_required
def generate_outfit():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId,'label':'Dress'}
    users_some_clothes = db.wardrobe.find(filter).limit(3)
    print("clothes, buth how many?")
    count = 0;

    clothes1 = users_some_clothes[1]
    clothes2 = users_some_clothes[2]
    clothes3 = users_some_clothes[3]
    print(clothes1)
    print(clothes2)
    print(clothes3)
    # for key, item in users_some_clothes:
    #     print(item['label'])
    # aici iau modelul si astept

    # option = request.form['options']
    # print(option)
    option = request.form.getlist('options')
    print(option)

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
    city1 = weather_data[0]
    city2 = weather_data[1]
    city3 = weather_data[2]

    result_weather = request.form.getlist('city')
    result_location = request.form.getlist('location')
    print(result_weather)
    print(result_location)


    return render_template('outfit_generator.html', outfit1 = clothes1, outfit2 = clothes2, outfit3= clothes3,
                           city1 = city1,city2 = city2, city3 = city3)


@app.route('/outfit/day', methods=['GET', 'POST'])
@login_required
def get_outfit():
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



    return render_template('outfit_of_the_day.html',weather_data = weather_data)


@app.route('/dashboard/',methods=['GET', 'POST'])
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

    return render_template('dashboard.html', weather_data = weather_data)


@app.route('/model', methods=['GET'])
@login_required
def index():
    # Main page
    return render_template('index.html')


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
        # _, color = predict_color(file_path)
        #
        # listToStr = ' '.join(map(str, color))
        # print(listToStr)
        # my_result = tuple(map(int, listToStr.split(' ')))
        # print(my_result)
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

    return render_template('wardrobe_all.html', wardrobes = users_clothes )


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join(
            'flaskapp/static/image_users/',secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        file_path_bd = os.path.join(
            '../static/image_users/', secure_filename(f.filename))

        # Make prediction
        preds = model_predict(file_path, model)
        _, color = predict_color(file_path)
        # path = ''
        # learn = load_learner(path, 'atr-recognition-stage-11-resnet34.pkl')
        # learn.show_results()

        listToStr = ' '.join(map(str, color))

        print(listToStr)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        userId = session['user']['_id']
        db.wardrobe.insert_one({ 'label': result, 'color': listToStr, 'userId':userId,
                               'file_path': file_path_bd })

        return result
    return None


@app.route('/color', methods=['GET', 'POST'])
@login_required
def def_color():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(
            'flaskapp/static/image_users/', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        file_path_bd = os.path.join(
            '../static/image_users/', secure_filename(f.filename))

        _, color = predict_color(file_path)

        listToStr = ' '.join(map(str, color))
        print(listToStr)
        # my_result = listToStr
        my_result = tuple(map(int, listToStr.split(' ')))
        print(my_result)

        return listToStr
    return None


if __name__ == '__main__':
    app.run(debug=True)
