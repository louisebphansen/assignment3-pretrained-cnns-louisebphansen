# Assignment 3: Using pretrained CNNs for image classification

This assignment is the third assignment for the course Visual Analytics on the elective Cultural Data Science at Aarhus University, 2023.

### Contributions

The code was created by me. However, inspiration was found on the dataset's Kaggle site (https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) and code from Session 09 from the Visual Analytics course was reused with some modifications. 

### Assignment description

- Write code which trains a classifier on this (*Indo-fashion*) dataset using a pretrained CNN like VGG16
- Save the training and validation history plots
- Save the classification report

### Contents of the repository
| Folder/File  | Contents| Description |
| :---:   | :---: | :--- |
|```out```|clf_report.txt, history_plot.png|The folder contains the output classification report and history plots from running the **indo_fashion_classifier.py** script with default arguments.|
|```src```|**data.py**, **indo_fashion_classifier.py**| The folder contains Python scripts for training a classifier on the indo-fashion dataset. **data.py** contains code to preprocess and prepare the indo-fashion dataset for a classification task. **indo_fashion_classifier.py** contains code to train a model and save classification report and history plot|
|README.md|-|Description and overview of repository, and how to use the code.|
|requirements.txt|-|Packages required to run the code.|
|run.sh|-|Bash script for running the indo-fashion classifier using default/predefined arguments|
|setup.sh|-|Bash script for setting up a virtual environment for the project|

### Data

The indo fashion dataset consists of more than a 100,000 colored images of Indian fashion clothes divided into 15 categories. The data is already split into training, validation and testing datasets, with 91,166 train images, 7,500 validation images, and 7,500 test images.

### Methods

*The following section describes the methods used in the provided Python scripts*

**Data preprocessing**

The preprocessing script, **data.py** uses Keras ImageDataGenerators to preprocess and prepare the data for the classifier by using a function which converts the images from RGB (*red-green-blue*) to BGR and zero-centers all color channels.
The ImageDataGenerators are then used in a Keras *flow_from_dataframe* pipeline, which generates batches of image data using information from a metadataframe, instead of relying heavily on your computer's memory by fetching all data at once. A seperate datagenerator-flow is applied to training, validation and testing datasets.

**Classification**

The **indo_fashion_classifier.py** script trains a convolutional neural network using a pretrained model, VGG16, to classify the 15 different classes of Indian fashion items. The classification layers of VGG16 are not used, as two new classification layers are added instead. The model uses ReLU as the activation function, stochastic gradient descent as the optimizer, and categorical cross-entropy as the loss function. The script creates and saves a classification report and model history plot in the ```out``` folder.

### Usage

All code for this assignment was designed to run on an *Ubuntu 22.10* operating system.

To reproduce the results in this repository, clone it using ```git clone```.

It is important that you run all scripts from the main folder, i.e., *assignment3-pretrained-cnns-louisebphansen* folder. Your terminal should look like this:

```
--your_path-- % assignment3-pretrained-cnns-louisebphansen % 
```

**Acquire the data**

Due to the size of the dataset (around 3GB), it is not uploaded to this repository. Instead, dowload the data here: https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset. When you download the folder, it will be called "archive" on your machine. Inside this folder, there is a folder called **images** and three metadata json files. Create a folder in the **images** folder called **metadata** and place the metadata files here. You should now have four sub-folders in the **images** folder called **metadata**, **train**, **val** and **test**. Next,  place the **images** folder in the main directory, i.e., inside **assignment3-pretrained-cnns-louisebphansen**! Your directory should look like this:

![Skærmbillede 2023-04-27 kl  18 08 29](https://user-images.githubusercontent.com/75262659/234921888-507bf1bf-15b3-454e-bc27-93394dc4d5e2.png)

#### Setup 
First, ensure that you have installed the *venv* package for Python (if not, run ```sudo apt-get update``` and ```sudo apt-get install python3-venv```). 

To set up the virtual environment, run ```bash setup.sh``` from the terminal. 

#### Run code
To run the classifier script, you can do the following:

**Run script with predefined arguments**
From the terminal (and still in the main folder), type ```bash run.sh``` which activates the virtual environment and runs the classifier script with the default arguments, i.e., uses two hidden layers of size 256 and 128, 10 epochs and saves the history plot as 'history_plot.png' abd classification report as 'clf_report.txt.


**Define arguments yourself**

First, activate the virtual environment, then run the script from the terminal with the desired arguments. Again, it is important that you run it from the main folder.


```
source env/bin/activate # activate virtual environment

python3 src/indo_fashion_classifier.py --first_layer_size <first_layer_size> --second_layer_size <second_layer_size> --epochs <epochs> --plot_name <plot_name> --report_name <report_name>
```

**Arguments**
- **first_layer_size:** Size of first classification layer. Default: 256
- **Second_layer_size:** Size of second classification layer. Default: 128
- **epochs**: Number of epochs to run the model for. Default: 10 
- **plot_name:** Name of the output history plot. Default: 'history_plot.png'
- **report_name:** Name of the output classification report. Default: 'clf_report.txt'


**NB**: Due to the size of the dataset, the code may take several hours to run, depending on the processing power of your machine.

### Results

![image](https://github.com/AU-CDS/assignment3-pretrained-cnns-louisebphansen/assets/75262659/f6e45e00-b4d6-443e-8523-f3090c93f8e0)

The history plots plotting training vs validation loss and training vs validation accuracy looks fine. There does not seem to be any sign of overfitting, as the two curves follow each other nicely. 

![image](https://github.com/AU-CDS/assignment3-pretrained-cnns-louisebphansen/assets/75262659/e1b54d29-0b43-4a22-b7ed-37db3ba5facf)

As seen from the classification report, the current model performs well with a 0.74 overall accuracy. However, when looking closer at the different categories, there is some variation in how well the model performs. It is apparantly very good at predicting blouses (F1: 0.92) and lehengas (F1: 0.86), but not as good at predicting dhoti pants (F1: 0.54) or gowns (F1: 0.47). 
