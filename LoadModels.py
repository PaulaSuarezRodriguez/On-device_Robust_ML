import sklearn.metrics
from keras.metrics import accuracy
from keras.saving.legacy.saved_model.serialized_attributes import metrics
from kiwisolver import BadRequiredStrength
from qkeras import *
# from typing import Any
import tensorflow as tf
from keras.models import load_model as a
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, \
    classification_report
from matplotlib import pyplot as plt
import seaborn as sns

# Setting random seed for ensuring reproducibility of results
seed = 42
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.semi_supervised import SelfTrainingClassifier
from keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version of LoadModels.py:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Reshape, Multiply, Concatenate, Add, DepthwiseConv2D, SeparableConv2D
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from PIL import Image
import os

# CODE FOR PREPROCESSING DATASET: heavily inspired by my supervisor's original code
# convert from color image (RGB) to grayscale
# source: opencv.org
# grayscale = 0.299*red + 0.587*green + 0.114*blue
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# Load data from TF Keras\n",
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() #keras built-in method
# input image dimensions
# we assume data format \"channels_last\
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]
# create saved_images folder\n",
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# convert color train and test images to gray\n",
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)
# display grayscale version of test images\n",
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
# CIFAR10 class names\n",
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

num_classes = len(class_names)

# Normalize pixel values to be between 0 and 1\n",
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], x_train_gray.shape[1], x_train_gray.shape[2], 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], x_test_gray.shape[1], x_test_gray.shape[2], 1)
# normalize input train and test grayscale images\n",
x_train_gray = x_train_gray.astype(np.float32) / 255
x_test_gray = x_test_gray.astype(np.float32) / 255
# Convert class vectors to binary class matrices.\n",
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
batch_size = 64
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
# input image dimensions\n",
input_shape = (x_train.shape[1], x_train.shape[2], 1)
input_shape_gray = (x_train_gray.shape[1], x_train_gray.shape[2], x_train_gray.shape[3])
x_train = (x_train * 255) - 128
x_test = (x_test * 255) - 128
x_train_gray = (x_train_gray * 255) - 128
x_test_gray = (x_test_gray * 255) - 128

# further split of training set into validation.This is an adaptation of my supervisor's original code from script setup_cifar.py
VALIDATION_SIZE = 1000
# To be used in model.fit, since the original code used x_test_gray, NOT x_test
x_validation_gray = x_train_gray[:VALIDATION_SIZE]
y_validation = y_train[:VALIDATION_SIZE]
x_train_gray = x_train_gray[VALIDATION_SIZE:]
y_train = y_train[VALIDATION_SIZE:]
x_validation = x_train[:VALIDATION_SIZE]
x_train = x_train[VALIDATION_SIZE:]

# Code for loading final models from the .keras files storing them, allowing us to evaluate their performance on test set.
reconstructed_model1 = tf.keras.models.load_model("ModelsKDv2/Jac_KD_with_Distiller_teacher.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_model2 = tf.keras.models.load_model("ModelsKDv2/Jac_KD_with_Distiller_student.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_model3 = tf.keras.models.load_model("ModelsKDv2/_best_tJac_teacher.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_model4 = tf.keras.models.load_model("ModelsKDv2/_best_vJac_teacher.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_model5 = tf.keras.models.load_model("ModelsKDv2/_best_tJac_student.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_model6 = tf.keras.models.load_model("ModelsKDv2/_best_vJac_student.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)

reconstructed_models = {"Jac_KD_with_Distiller_teacher": reconstructed_model1,
                        "Jac_KD_with_Distiller_student": reconstructed_model2,
                        "_best_tJac_teacher": reconstructed_model3,
                        "_best_vJac_teacher": reconstructed_model4,
                        "_best_tJac_student": reconstructed_model5,
                        "_best_vJac_student": reconstructed_model6}

test_accuracies = {}
train_accuracies = {}
val_accuracies = {}
for reconstructed_model_name, reconstructed_model in reconstructed_models.items():
    config = reconstructed_model.get_config()
    weights = reconstructed_model.get_weights()
    # summary = reconstructed_model.summary() # permanently commented out, not needed in latest version.

    if reconstructed_model == reconstructed_model5:
        reconstructed_model.compile(
            metrics=['accuracy'])  # in the case of student's b_v and b_t since they are CUSTOM ModelCheckpoints

    if reconstructed_model == reconstructed_model6:
        reconstructed_model.compile(
            metrics=['accuracy'])  # in the case of student's b_v and b_t since they are CUSTOM ModelCheckpoints

    test_accuracy = reconstructed_model.evaluate(x_test_gray, y_test, verbose=0)
    train_accuracy = reconstructed_model.evaluate(x_train_gray, y_train, verbose=0)
    val_accuracy = reconstructed_model.evaluate(x_validation_gray, y_validation, verbose=0)

    print(f"Model: {reconstructed_model_name}")

    print(f"\nConfiguration of {reconstructed_model_name}: {config}")
    print(f"\nWeights of {reconstructed_model_name}: \n {weights}")
    print(f"\nSummary of {reconstructed_model_name}:")
    print(
        f"{reconstructed_model.summary()}")
    print(f"\nTEST LOSS and TEST ACCURACY of loaded model {reconstructed_model_name}: {test_accuracy}")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)
    test_accuracies.update({reconstructed_model_name: test_accuracy})
    train_accuracies.update({reconstructed_model_name: train_accuracy})
    val_accuracies.update({reconstructed_model_name: val_accuracy})

# Student was trained with "_best_vJac_teacher.keras". As such, level of compression achieved between teacher and student
# is revealed by comparing "_best_vJac_teacher.keras" with whichever is the best performing saved student model out of
# "Jac_KD_with_Distiller_student", "_best_tJac_student" and "_best_vJac_student.keras". We already know from results achieved
# that "_best_vJac_student.keras" is that model so, level of compression achieved, further proving KD did take place rather
# successfully (at least by one standard (compression) out of 2 (compression and barely any drop in clean accuracy))
# = ((teacher_params - student_params) / teacher_params) * 100.

compression_level = (((reconstructed_models.get("_best_vJac_teacher").count_params() - reconstructed_models.get("_best_vJac_student").count_params()) / reconstructed_models.get("_best_vJac_teacher").count_params() * 100))
print(f"Compression level achieved by Knowledge Distillation (difference in the number of parameters between the teacher "
      f"model used to train the student and best-performing student model saved) = {compression_level} % \n")

accuracy_difference = ((test_accuracies.get("_best_vJac_teacher")[1] - test_accuracies.get("_best_vJac_student")[1]) * 100)
print(f"Test Accuracy difference between the teacher model used to train the student and best-performing student model saved) = {accuracy_difference} % \n")

print(f"Summary of test loss and test accuracies by each model evaluated: {test_accuracies}")
print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)


# Functions generating confusion matrices, bar graphs and Classification Reports for illustrating accuracy results and relevant comparisons made in the
# Results Chapter of the Final Report (libraries matplotlib and sklearn were used for this purpose):

def accuracy_comparison_teachers_bar_graph(test_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best teacher\nmodel as per\ntraining loss', 'Best teacher\nmodel as per\nvalidation loss', 'Teacher model\nresulting from\nlast epoch']
    accuracies = [test_accuracies.get("_best_tJac_teacher")[1], test_accuracies.get("_best_vJac_teacher")[1], test_accuracies.get("Jac_KD_with_Distiller_teacher")[1]]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, accuracies)
    plt.title('Test Accuracies of Teacher Models Saved:\nWhich is the best teacher model overall?', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.xlabel('\nTrained Teacher Models', fontsize=20)
    plt.ylim(0.80, 0.8125)  #works
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print("Title of bar graph: Test Accuracies of Teacher Models Saved: Which is the best teacher model overall?")
    print(f"\nTest Accuracy of _best_tJac_teacher: {(accuracies[0])*100} \nTest Accuracy of _best_vJac_teacher: {(accuracies[1])*100}"
          f"\nTest Accuracy of Jac_KD_with_Distiller_teacher: {(accuracies[2])*100}")
    print("Difference between _best_vJac_teacher and _best_tJac_teacher Test Accuracy =", accuracies[1]-accuracies[0])
    print("Difference between _best_vJac_teacher and Jac_KD_with_Distiller_teacher Test Accuracy =", accuracies[1]-accuracies[2])
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def overfitting_test_teachers_bar_graph(train_accuracies, val_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best teacher\nmodel as per\ntraining loss', 'Best teacher\nmodel as per\nvalidation loss', 'Teacher model\nresulting from\nlast epoch']
    plt.figure(figsize=(10, 6))
    train=[train_accuracies.get("_best_tJac_teacher")[1], train_accuracies.get("_best_vJac_teacher")[1], train_accuracies.get("Jac_KD_with_Distiller_teacher")[1]]
    validation=[val_accuracies.get("_best_tJac_teacher")[1], val_accuracies.get("_best_vJac_teacher")[1], val_accuracies.get("Jac_KD_with_Distiller_teacher")[1]]
    plt.bar(np.arange(len(bars)) - 0.4, train, label='When predicting on training dataset', width=0.4, align='edge')
    plt.bar(np.arange(len(bars)), validation, label='When predicting on validation dataset', width=0.4, align='edge')
    plt.title('Training and Validation Accuracies of Teacher Models:\nTesting for overfitting', fontsize=20)
    plt.xlabel('\nTrained Teacher Models', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim(0.75, 0.89)
    plt.xticks(np.arange(len(bars)), bars, fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print("Title of bar graph: Training and Validation Accuracies of Teacher Models: Testing for overfitting")
    print(f"\nTraining Accuracy of _best_tJac_teacher: {(train[0])*100} \nValidation Accuracy of _best_tJac_teacher: {(validation[0])*100}"
          f"\nDifference between _best_tJac_teacher Training Accuracy and Validation Accuracy = {(train[0]-validation[0])*100}")

    print(f"\nTraining Accuracy of _best_vJac_teacher: {train[1]} \nValidation Accuracy of _best_vJac_teacher: {(validation[1])*100}"
          f"\nDifference between _best_vJac_teacher Training Accuracy and Validation Accuracy = {(train[1]-validation[1])*100}")

    print(f"\nTraining Accuracy of Jac_KD_with_Distiller_teacher: {(train[2])*100} \nValidation Accuracy of Jac_KD_with_Distiller_teacher: {(validation[2])*100}"
          f"\nDifference between Jac_KD_with_Distiller_teacher Training Accuracy and Validation Accuracy = {(train[2]-validation[2])*100}")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def overfitting_test_best_teacher_bar_graph(train_accuracies, val_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['When predicting on\ntraining dataset', 'When predicting on\nvalidation dataset']
    accuracies = [train_accuracies.get("_best_vJac_teacher")[1], val_accuracies.get("_best_vJac_teacher")[1]]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, accuracies)
    plt.title('Training and Validation Accuracies of Best Teacher Model: \nNo overfitting', fontsize=20)
    plt.xlabel('\nBest teacher model overall', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim(0.75, 0.87)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    print("Title of bar graph: Training and Validation Accuracies of Best Teacher Model: No overfitting")
    print(f"\nTraining Accuracy of _best_vJac_teacher: {(accuracies[0])*100} \nValidation Accuracy of _best_vJac_teacher: {(accuracies[1])*100}"
          f"\nDifference between _best_vJac_teacher Training Accuracy and Validation Accuracy = {(accuracies[0]-accuracies[1])*100}")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def accuracy_comparison_students_bar_graph(test_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best student\nmodel as per\ntraining accuracy', 'Best student\nmodel as per\nvalidation accuracy', 'Student model\nresulting from\nlast epoch']
    accuracies = [test_accuracies.get("_best_tJac_student")[1], test_accuracies.get("_best_vJac_student")[1], test_accuracies.get("Jac_KD_with_Distiller_student")[1]]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, accuracies)
    plt.title('Test Accuracies of Student Models Saved:\n(all trained via KD using best teacher model)\nWhich is the best student model overall?', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.xlabel('\nDistilled Student Models', fontsize=20)
    plt.ylim(0.75, 0.80)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print("Title of bar graph: Test Accuracies of Student Models Saved (all trained via KD using best teacher model): Which is the best student model overall?")
    print(f"\nTest Accuracy of _best_tJac_student: {(accuracies[0])*100} \n Test Accuracy of _best_vJac_student: {(accuracies[1])*100}"
          f"\n Test Accuracy of Jac_KD_with_Distiller_student: {(accuracies[2])*100}")
    print("Difference between _best_vJac_student and _best_tJac_student Test Accuracy =", (accuracies[1]-accuracies[0])*100)
    print("Difference between _best_vJac_student and Jac_KD_with_Distiller_student Test Accuracy =", (accuracies[1]-accuracies[2])*100)
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def overfitting_test_students_bar_graph(train_accuracies, val_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars = ['Best student\nmodel as per\ntraining accuracy', 'Best student\nmodel as per\nvalidation accuracy', 'Student model\nresulting from\nlast epoch']
    plt.figure(figsize=(10, 6))
    train=[train_accuracies.get("_best_tJac_student")[1], train_accuracies.get("_best_vJac_student")[1], train_accuracies.get("Jac_KD_with_Distiller_student")[1]]
    validation=[val_accuracies.get("_best_tJac_student")[1], val_accuracies.get("_best_vJac_student")[1], val_accuracies.get("Jac_KD_with_Distiller_student")[1]]
    plt.bar(np.arange(len(bars)) - 0.4, train, label='When predicting on\ntraining dataset', width=0.4, align='edge')
    plt.bar(np.arange(len(bars)), validation, label='When predicting on\nvalidation dataset', width=0.4, align='edge')
    plt.title('Training and Validation Accuracies of Student Models:\nTesting for overfitting', fontsize=20)
    plt.xlabel('\nTrained Student Models', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim(0.75, 0.85)
    plt.xticks(np.arange(len(bars)), bars, fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print("Title of bar graph: Training and Validation Accuracies of Student Models: Testing for overfitting")
    print(f"\nTraining Accuracy of _best_tJac_student: {train[0]*100} \n Validation Accuracy of _best_tJac_student: {(validation[0])*100}"
          f"\n Difference between _best_tJac_student Training Accuracy and Validation Accuracy = {(train[0]-validation[0])*100}")

    print(f"\nTraining Accuracy of _best_vJac_student: {train[1]*100} \n Validation Accuracy of _best_vJac_student: {validation[1]*100}"
          f"\n Difference between _best_vJac_student Training Accuracy and Validation Accuracy = {(train[1]-validation[1])*100}")

    print(f"\nTraining Accuracy of Jac_KD_with_Distiller_student: {train[2]*100} \n Validation Accuracy of Jac_KD_with_Distiller_student: {validation[2]*100}"
          f"\n Difference between Jac_KD_with_Distiller_student Training Accuracy and Validation Accuracy = {(train[2]-validation[2])*100}")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def overfitting_test_best_student_bar_graph(train_accuracies, val_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['When predicting on\ntraining dataset', 'When predicting on\nvalidation dataset']
    accuracies = [train_accuracies.get("_best_vJac_student")[1], val_accuracies.get("_best_vJac_student")[1]]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, accuracies)
    plt.title('Training and Validation Accuracies of Best Distilled Student Model:\nNo overfitting', fontsize=20)
    plt.xlabel('\nBest Distilled Student Model overall', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim(0.75, 0.85)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    print("Title of bar graph: Training and Validation Accuracies of Best Distilled Student Model: No overfitting")
    print(f"\nTraining Accuracy of _best_vJac_student: {accuracies[0]*100} \n Validation Accuracy of _best_vJac_student: {accuracies[1]*100}"
          f"\n Difference between _best_vJac_student Training Accuracy and Validation Accuracy = {(accuracies[0]-accuracies[1])*100}")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def compression_level_bar_graph(reconstructed_models): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best Teacher\n(used in training of the student)', 'Best Distilled student']
    parameter_count = [reconstructed_models.get("_best_vJac_teacher").count_params(), reconstructed_models.get("_best_vJac_student").count_params()]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, parameter_count)
    plt.title('Compression level achieved by Knowledge Distillation', fontsize=20)
    plt.xlabel('\nKD Models', fontsize=20)
    plt.ylabel('Number of parameters', fontsize=20)
    #plt.ylim(600000, 2000000) # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylim.html
    #plt.ylim(600000, 1500000) #tested better than one above
    plt.ylim(600000, 1200000) # testing
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    print("Title of bar graph: Compression level achieved by Knowledge Distillation")
    print(f"Number of parameters of _best_vJac_teacher = {parameter_count[0]} \nNumber of parameters of _best_vJac_student = {parameter_count[1]}")
    print(f"Difference in the Number of Parameters between _best_vJac_teacher and _best_vJac_student = {(parameter_count[0]-parameter_count[1])}")
    print(f"Compression_percentage = {((parameter_count[0] - parameter_count[1]) / parameter_count[0]) * 100} %")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def accuracy_comparison_KD_models_bar_graph(test_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best teacher\n(used in training of the student)', 'Best distilled student']
    accuracies = [test_accuracies.get("_best_vJac_teacher")[1], test_accuracies.get("_best_vJac_student")[1]]
    plt.figure(figsize=(10, 6))
    plt.bar(bars, accuracies)
    plt.title('Test accuracies of teacher and student model:\nLess than 5% drop in accuracy', fontsize=20) #Definite one to be used for ModelsKDv2
    plt.xlabel('\nKD Models', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.ylim(0.75, 0.85) #Definite one to be used for ModelsKDv2
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    print("Title of bar graph: Test accuracies of teacher and student model: Less than 5% drop in accuracy")
    print(f"\nTest Accuracy of _best_vJac_teacher: {accuracies[0]*100} \n Test Accuracy of _best_vJac_student: {accuracies[1]*100}")
    print(f"Difference between _best_vJac_teacher and _best_vJac_student Test Accuracy = {(accuracies[0]-accuracies[1])*100} %")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def accuracy_comparison_student_models_bar_graph(test_accuracies): # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    bars=['Best student\nmodel as per\ntraining accuracy', 'Best student\nmodel as per\nvalidation accuracy', 'Student model\nresulting from\nlast epoch']
    plt.figure(figsize=(10, 6))
    test_teach=test_accuracies.get("_best_vJac_teacher")[1]
    test_students=[test_accuracies.get("_best_tJac_student")[1], test_accuracies.get("_best_vJac_student")[1], test_accuracies.get("Jac_KD_with_Distiller_student")[1]]
    plt.bar(np.arange(len(bars)) - 0.4, test_teach, label='Test Accuracy of Best teacher (used in training of the students)', color="red", width=0.4, align='edge')
    plt.bar(np.arange(len(bars)), test_students, label='Test Accuracy of Student', color="purple", width=0.4, align='edge')
    plt.title('Test Accuracy of Teacher\nvs\nTest Accuracy of each saved Student model', fontsize=20)
    plt.xlabel('\nTrained Student Models', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.ylim(0.75, 0.83)
    plt.xticks(np.arange(len(bars)), bars, fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    print("Title of bar graph: Test Accuracy of Teacher vs Test Accuracy of each saved Student model")
    print(f"\nTest Accuracy of _best_vJac_teacher: {(test_teach)*100} \nTest Accuracy of _best_tJac_student: {(test_students[0])*100} %"
          f"\nDifference between Test Accuracies of _best_vJac_teacher and _best_tJac_student = {(test_teach-test_students[0])*100} %")

    print(f"\nTest Accuracy of _best_vJac_teacher: {(test_teach) * 100} \nTest Accuracy of _best_vJac_student: {(test_students[1]) * 100} %"
        f"\nDifference between Test Accuracies of _best_vJac_teacher and _best_vJac_student = {(test_teach - test_students[1]) * 100} %")

    print(f"\nTest Accuracy of _best_vJac_teacher: {(test_teach) * 100} \nTest Accuracy of Jac_KD_with_Distiller_student: {(test_students[2]) * 100} %"
        f"\nDifference between Test Accuracies of _best_vJac_teacher and Jac_KD_with_Distiller_student = {(test_teach - test_students[2]) * 100} %")
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)

def confusion_matrix_test_set_teacher(reconstructed_models): # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    y_pred = np.argmax(reconstructed_models.get("_best_vJac_teacher").predict(x_test_gray), axis=1)  # trained student predictions
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred) #, labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    cm_disp.plot()
    plt.title("Confusion Matrix of the Teacher Model's\npredictions on the test dataset")
    plt.show()

def confusion_matrix_test_set_student(reconstructed_models): # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    y_pred = np.argmax(reconstructed_models.get("_best_vJac_student").predict(x_test_gray), axis=1)  # trained student predictions
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred) #, labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    cm_disp.plot()
    plt.title("Confusion Matrix of the Student Model's\npredictions on the test dataset")
    plt.show()

def classification_report_best_teacher(reconstructed_models):
    y_pred = np.argmax(reconstructed_models.get("_best_vJac_teacher").predict(x_test_gray), axis=1)  # teacher predictions
    y_true = np.argmax(y_test, axis=1)
    print("\n Classification Report of the Teacher Model: \n", classification_report(y_true, y_pred))
    classification_report(y_true, y_pred)

def classification_report_best_student(reconstructed_models):
    y_pred = np.argmax(reconstructed_models.get("_best_vJac_student").predict(x_test_gray), axis=1)  # trained student predictions
    y_true = np.argmax(y_test, axis=1)
    print("\n Classification Report of the Student Model: \n", classification_report(y_true, y_pred))
    classification_report(y_true, y_pred)

if __name__ == "__main__":
    accuracy_comparison_teachers_bar_graph(test_accuracies)
    overfitting_test_teachers_bar_graph(train_accuracies, val_accuracies)
    overfitting_test_best_teacher_bar_graph(train_accuracies, val_accuracies)
    accuracy_comparison_students_bar_graph(test_accuracies)
    overfitting_test_students_bar_graph(train_accuracies, val_accuracies)
    overfitting_test_best_student_bar_graph(train_accuracies, val_accuracies)
    compression_level_bar_graph(reconstructed_models)
    accuracy_comparison_KD_models_bar_graph(test_accuracies)
    accuracy_comparison_student_models_bar_graph(test_accuracies)
    confusion_matrix_test_set_teacher(reconstructed_models)
    confusion_matrix_test_set_student(reconstructed_models)
    # roc_curve_display_best_student(reconstructed_models) # only for binary classifier. so, discarded
    classification_report_best_teacher(reconstructed_models)
    classification_report_best_student(reconstructed_models)

K.clear_session()
print("Session cleared")  # Clear memory