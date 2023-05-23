''' 
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 3: Pretrained CNN's

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains code to train a classifier on the 'indo-fashion' dataset.
Running the script saves a classification report and history plot in the 'out' folder of this repository.

'''

# generic tools
import os
import pandas as pd
import numpy as np
import argparse

# tf tools
import tensorflow as tf

# tf image tools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 VGG16)

# tf model tools
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# scikit-learn
from sklearn.metrics import classification_report

# for plotting
import matplotlib.pyplot as plt

# define argument parser
def argument_parser():

    # define parser
    ap = argparse.ArgumentParser()

    # add arguments
    ap.add_argument("--first_layer_size", type=int, help="Size of first classification layer", default=256)
    ap.add_argument("--second_layer_size", type=int, help="Size of second classification layer", default=128)
    ap.add_argument("--epochs", type=int, help = "Number of epochs", default=10)
    ap.add_argument("--plot_name", help="Name of the output history plot", default='history_plot.png')
    ap.add_argument("--report_name", help="Name of the output classification report", default='clf_report.txt')
    args = vars(ap.parse_args())
    
    return args

def build_model(first_layer_size, second_layer_size):
    '''
    Builds a convolutional neural network using the pretrained VGG16 model as feature extractor. Model has two classification layers and a final output layer.
    Code is borrowed and adapted from the Session 9 notebook of the Visual Analytics course @ Aarhus University, 2023.
    
    Returns:
    A compiled model that can be fit and used for a classification task.
    
    '''
    
    # load model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(first_layer_size, 
               activation='relu')(bn)
    class2 = Dense(second_layer_size, 
               activation='relu')(class1)
    output = Dense(15, 
               activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
              outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model
 
def save_plot_history(H, epochs, name):
    '''
    Saves the validation and loss history plots of a fitted model in the 'out' folder.
    Code is borrowed and adapted from the Session 9 notebook of the Visual Analytics course @ Aarhus University, 2023.
    
    Arguments:
    - H: Saved history of a model fit
    - epochs: Number of epochs the model runs on
    - name: What the plot should be called
    
    Returns:
        None

    '''
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join('out', name))

def create_report(model, test_gen, filename):
    '''
    Predicts on test data using model and saves the classification report in the 'out' folder.
    
    Arguments:
    - Model: a trained model
    - test_gen: Keras DataFrameIterator with test data (i.e., shuffle must be False)
    - filename: name of the saved classification report

    Returns:
        None
    '''

    # predict test data using model
    pred = model.predict(test_gen)
    predicted_classes = np.argmax(pred,axis=1)

    # reset test generator to start from first batch (to match outputs to predicted data)
    test_gen.reset()

    # get true labels of test data
    y_true = test_gen.classes

    labels = list(test_gen.class_indices.keys())

    # create classification report from predicted and true labels
    report = classification_report(y_true,
                            predicted_classes, target_names = labels)
    
    # save report
    out_path = os.path.join("out", filename)

    with open(out_path, 'w') as file:
                file.write(report)

def main():

    # parse arguments
    args = argument_parser()

    # import data generation function from data script
    from data import prep_data

    # create generators
    train_gen, val_gen, test_gen = prep_data()

    # build model with desired layer sizes
    model = build_model(args['first_layer_size'], args['second_layer_size'])

    # fit model and save history
    H = model.fit_generator(generator=train_gen, # fit model with generators
                    steps_per_epoch=128, 
                    validation_data=val_gen,
                    validation_steps=128,
                    epochs=args['epochs'])

    # save plot
    save_plot_history(H, args['epochs'], args['plot_name'])

    # save report
    create_report(model, test_gen, args['report_name'])

if __name__ == '__main__':
   main()

