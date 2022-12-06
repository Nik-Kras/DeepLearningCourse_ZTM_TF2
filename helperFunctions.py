import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import matplotlib.image as mpimg    # To view image
import numpy as np
import datetime
import random
import os

def print_tf_version():
    """
    Prints current TensorFlow version
    To check whether you are using TF2
    """
    v = tf.__version__
    if v[0] == '2':
        print("Apropriate version!")
    print("Your version is: ", v)

##############################################
### 03. Classification  ######################
##############################################

def plot_2D_dots_predictions(train_data,
                             test_data,
                             train_labels,
                             test_labels,
                             predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()


def plot_2D_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
    1. CS231n - https://cs231n.github.io/neural-networks-case-study/
    2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

##############################################
### 04. Computer Vision  #####################
##############################################

def check_files_and_directories(folder_path="pizza_steak"):
    """
    Walks through all folders inside the `folder_path`
    and prints number of files it found

    Use it to ivestigate how many training / testing
    examples you have in dataset
    """
    for root, dirs, files in os.walk(folder_path):
        print(f"There are #{len(dirs)} directories and #{len(files)} files In the directory {root}")

def view_random_image_from_dataset(target_dir, target_class):
    """
    Prints a random image with a given class from a dataset.
    It is assumed that `target_dir` will contain folders for each class,
    where `target_class` is one of the name
    """

    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape: {img.shape}") # show the shape of the image

    return img

def load_scale_and_reshape_image(img_path, img_size=224):
    """
    The function takes a path to an image and returns an image in a 
    Tensor format with proper scaling and reshaped for a model
    """
    # Load an image
    img = tf.io.read_file(img_path)
    # Decode an image to tensor and ensure 3 channels
    img = tf.image.decode_image(img, channels=3)
    # Resize an image
    img = tf.image.resize(img, size=[img_size, img_size])
    # Scale an image
    img = img / 255.

    return img

def plot_loss_curves_train_validation(my_history):
    """
    Plots separate loss curves figures for training and validation metrics
    """
    loss = my_history.history["loss"]
    accuracy = my_history.history["accuracy"]
    val_loss = my_history.history["val_loss"]
    val_accuracy = my_history.history["val_accuracy"]

    epochs = range(len(loss))  # Length of any training metric gives the number of epochs

    # Plot Loss
    plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss,     label = "Training Loss"  )
    plt.plot(epochs, val_loss, label = "Validation Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy,     label = "Training Accuracy"  )
    plt.plot(epochs, val_accuracy, label = "Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

def binary_predict_and_plot(model, img_path, class_names):
    """
    It loads an image, makes preprocessing
    Using a model makes a prediction of a class of a model
    And plots an image with assigned class and 
    Probability of it for a binary classification
    """
    img = load_scale_and_reshape_image(img_path)
    img_pass = tf.expand_dims(img, axis=0)
    probability = model.predict(img_pass)
    pred_class = class_names[int(tf.round(probability))]
    if probability < 0.5:
        probability = 1 - probability
    plt.figure()
    plt.imshow(img)
    plt.title(pred_class + " " + "{:.1f}".format(100*probability[0][0]) + "%")
    plt.axis(False)

##############################################
### 05. Transfer Learning: feature extraction 
##############################################

def view_random_image(data_type=None, classes=None, n_train=75, n_test=250):
    """
    The function views random image from a given set of classes and set of datatypes
    Names of `classes` and `data_type` must be the same as in folders tree

    :param data_type: list of parents folder of classes. Use `None` to apply the value by default [\"train\", \"test\"]. 
    :param classes: list of folders representing classes in dataset. Looks like: [\"hamburger\", \"pizza\", ..., \"grilled_salmon\"]
    :param N_train: number of one class examples in Training folder
    :param N_test: number of one class examples in Testing folder
    """

    option_type = data_type
    if data_type is None:
        option_type = ["train", "test"]

    option_class = classes
    if classes is None:
        option_class = ["hamburger", "pizza", "ice_cream", "ramen", "chicken_curry",
                    "steak", "chicken_wings", "sushi", "fried_rice", "grilled_salmon"]

    chose_type = random.choice(option_type)
    chose_class = random.choice(option_class)
    n_images = n_test-1
    if chose_type == "train": n_images = n_train-1
    chose_image_n = random.randint(0, n_images)

    path = "10_food_classes_10_percent/"+chose_type+"/"+chose_class
    images_file = os.listdir(path)
    random_image_file = images_file[chose_image_n]

    img = mpimg.imread(path + "/" + random_image_file)
    plt.imshow(img)
    plt.title(chose_class)
    plt.axis("off")
    plt.show()

    print(f"Image shape: {img.shape}") # show the shape of the image
    

def create_tensorboard_callback(dir_name: str, experiment_name: str):
    """
    Creates a callback to save training logs into TensorBoard

    :param dir_name: path or name of a folder to save logs
    :param experiment_name: name of the folder for a current model training to be saved separately in the Log folder
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

def create_feature_extracting_classifier_model(model_url: str, n_units: int, n_classes: int, input_shape):
    """
    Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    :param model_url: A TensorFlow Hub feature extraction URL. 
        This extractor is added as first layers of the model
    :param n_units: Number of hidden neurons after feature extractor.
    :param n_classes: Number of output neurons, should be equal to number of target classes.

    Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
    """
    # Transfer feature extractor
    feature_extraction = hub.KerasLayer(model_url,        # Specify the model
                                        trainable=False,  # Freeze trained patterns
                                        input_shape=input_shape, # Set correct input shape
                                        name="Feature_Extractor")
    # Create a model
    model = tf.keras.Sequential([
        feature_extraction,
        layers.Dense(n_units, activation="relu"),
        layers.Dense(n_classes, activation="sigmoid")
    ])

    return model