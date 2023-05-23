''' 
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 3: Pretrained CNN's

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains code to load and preprocess the 'indo-fashion' dataset.
The script creates DataGenerators using flow_from_dataframe to generate batches of image data, preparing it for a classifier.

'''
# import packages
import os
import pandas as pd

# tf tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


def create_generator(directory, json_name, size, shuffle):
    '''
    Creates datagenerators that preprocesses the data and applies it to data fetched from a json metadataframe
    
    Arguments:
    - directory: Path to main directory
    - json_name: Name of json file containing the metadata. Must be a subfolder of the "images" folder and must contain metadata json files.
    - size: Desired size of the images
    - shuffle: Whether to shuffle the data or not
    
    Returns:
    A Keras DataFrameIterator

    '''
    df = pd.read_json(os.path.join("images", "metadata", json_name), lines=True) # convert json data into a dataframe

    # create datagenerator
    datagenerator = ImageDataGenerator(
        preprocessing_function = preprocess_input) # use VGG's preprocessing function

    # fetch data from dataframe
    gen = datagenerator.flow_from_dataframe(
        dataframe=df, # metadataframe
        directory=directory, # main directory
        x_col='image_path', # column containing the absolute path to each image
        y_col='class_label', # column containing the class label of each image
        batch_size=32, 
        shuffle=shuffle, # whether to shuffle the data
        class_mode="categorical", # type of class
        target_size=(size,size)) # size of image
    
    return gen

def prep_data():
    '''
    Creates Keras DataFrameIterators from training, validation and testing metadataframes.

    Returns:
        Three DataFrameIterators
    '''
    
    directory = os.getcwd() # get current directory (i.e., main folder)

    # create training and validation generators. Shuffle is set to true
    train_gen = create_generator(directory, "train_data.json", 224, True)
    val_gen = create_generator(directory, "val_data.json", 224, True)

    # create test generator. Shuffle is set to false to get outputs in the right order
    test_gen = create_generator(directory, "test_data.json", 224, False)

    return train_gen, val_gen, test_gen