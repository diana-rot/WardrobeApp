from __future__ import absolute_import, division, print_function, unicode_literals
# install helper utilities
import ssp19ai_utils.utils as utils
import importlib
from numpy import mean
importlib.reload(utils)
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle #for evaluating
import numpy as np

# TensorFlow and tf.keras
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf # The framework to run our models
from tensorflow import keras # High order layers, models, etc
from tensorflow.keras.utils import to_categorical # Utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
# from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version is " + tf.__version__)


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    unique = unique_labels(y_true, y_pred)
    if(classes is None):
      classes = unique
    else:
      classes = classes[unique]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def label_with_highest_prob(probabilities):
  return np.argmax(probabilities, axis = 1)


def load_dataset():
    # load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # reshape dataset to have a single channel
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    # one hot encode target values
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels, class_names


# scale pixels
# normalize the values
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    #train images
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images

    return train_norm, test_norm


def plot_single_image_correct(i, predictions, true_labels, images, class_names=None, cmap=plt.cm.binary):
    predictions_array, true_label, img = predictions[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=cmap)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    class_name = true_label if class_names is None else class_names[true_label]
    class_name_predicted = predicted_label if class_names is None else class_names[predicted_label]
    plt.xlabel("{} {:2.0f}% ({})".format(class_name_predicted,
                                         100 * np.max(predictions_array),
                                         class_name),
               color=color)


def plot_value_array(i, predictions, true_labels):
    predictions_array, true_label = predictions[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_multi_images_prob(predictions, labels, images, class_names=None, start=0, num_rows=5, num_cols=3, cmap=plt.cm.binary ):
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    index = i + start
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_single_image_correct(index, predictions, labels, images, class_names, cmap=cmap)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(index, predictions, labels)
  plt.show()

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     activation='relu', kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
   # utils.draw_model(model)
   #  model.save('my_new_my_model.h5')

    return model



def evaluate_model(dataX, dataY,  class_names, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits

    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        train_images, train_labels, test_images, test_labels =\
            dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(train_images, train_labels, epochs=4,
                            batch_size=32, validation_data=(test_images, test_labels),
                            verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_images, test_labels, verbose=0)
        # utils.plot_accuracy_and_loss(history)
        print('> %.3f' % (acc * 100.0))
        # append scores
        scores.append(acc)
        histories.append(history)

    model.save('my_model.h5')
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    test_images = test_images / 255.0

    predictions = model.predict(test_images)
    print(predictions)
    predicted_classes = label_with_highest_prob(predictions)
    print(predicted_classes)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Plot the matrix


    plot_confusion_matrix(y_pred=predicted_classes,
                            y_true=test_labels,
                            classes=np.array(class_names))
    plt.show()

    plot_multi_images_prob(predictions, test_labels, test_images)


    #
    # # visualitzation of corectly predicted classes
    corect_predictions = []
    for i in range(len(test_labels)):
        if(predicted_classes[i] == test_labels[i]):
            corect_predictions.append(i)
        if(len(corect_predictions) == 25):
            print("correctly_predicted")
            break

    row = 5
    column = 5
    fig, ax = plt.subplots(5, 5, figsize=(10, 5))
    fig.set_size_inches(8, 8)
    j = 0;
    for aux in range(0, 5):
        for auxi in range(0, 5):

            ax[aux, auxi].imshow(test_images[corect_predictions[j]].reshape(28, 28), cmap='gray')
            print(aux,auxi,j)
            ax[aux, auxi].set_title(
                "P:" + str(class_names[predicted_classes[corect_predictions[j]]]) + " " + "A: " +
                str(class_names[test_labels[corect_predictions[j]]]), fontsize=7)
            j= j+1

    plt.show();

    # visualitzation of incorectly predicted classes

    incorrect_predictions = []
    for j in range(len(test_labels)):
        if (not predicted_classes[j] == test_labels[j]):
            incorrect_predictions.append(j)
        if (len(incorrect_predictions) == 25):
            print("hello")
            break

    fig, ax = plt.subplots(5, 5, figsize=(10, 5))
    fig.set_size_inches(8, 8)
    j = 0;
    for aux in range(0, 5):
        for auxi in range(0, 5):

            ax[aux, auxi].imshow(test_images[incorrect_predictions[j]].reshape(28, 28), cmap='gray')
            print(aux,auxi,j)
            ax[aux, auxi].set_title(
                "P:" + str(class_names[predicted_classes[incorrect_predictions[j]]]) + " " + "A: " +
                str(class_names[test_labels[incorrect_predictions[j]]]),  fontsize=7)
            j= j+1

    plt.show();



    return scores, histories

# run the test harness for evaluating a model
def summarize_diagnostics(histories):


    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    model = define_model()
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    predictions = model.predict(test_images)
    print(predictions)

    predicted_classes = label_with_highest_prob(predictions)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Plot the matrix

    plot_confusion_matrix(y_pred=predicted_classes,
                            y_true=test_labels,
                            classes=np.array(class_names))
    plt.show()

    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'],
                    color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'],
                    color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'],
                    color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'],
                    color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()

def run_test_harness():

    # load dataset
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images, train_labels, test_images, test_labels, class_names= load_dataset()
    # prepare pixel data
    train_images, test_images = prep_pixels(train_images, test_images)
    # evaluate model

    scores, histories = evaluate_model(train_images, train_labels, class_names)
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


if __name__ == "__main__":
    run_test_harness()
