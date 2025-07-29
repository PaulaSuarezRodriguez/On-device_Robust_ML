'''
This version takes inspiration from "Keras 2 KD Recipes" (https://keras.io/examples/keras_recipes/better_knowledge_distillation/#distillation-utility)
used in Train_QJacobian_KD_with_Distiller_FourthAttempt(Definite).py but with a Distiller class similar to that of
Train_QJacobian_KD_with_Distiller_FirstAttempt(Fail).py

For tracking and saving the best teacher model, I have used Keras built-in ModelCheckpoints and passed them as arguments
to callback in the fit method for the teacher. For tracking and saving the best student model, I have used my own custom
class CustomModelCheckpoints and passed them as arguments to callback in the fit method for the student; the reason for
this being since student, even though a Keras Model and therefore compatible with Keras built-in ModelCheckpoints, it is
trained within another (also, custom) class Distiller, using ModelCheckpoints causes a "DictWrapper error even when the
ModelCheckpoints to be passed to the callbacks parameter of the fit method for the student are defined INSIDE the
Distiller class". Additionally, also because of the custom class Distiller, the metrics for monitoring the
CustomModelCheckpoints of the student can NOT be loss or val_loss, and so I have changed them to val_accuracy and
accuracy. The reason for this being, even though my class CustomModelCheckpoint is a subclass of Keras ModelCheckpoint
and Distiller is a subclass of Keras Model, the overriding of class compute_loss in Distiller means the models that were
inherited from Keras Models, namely that calculating student loss (which given the nature of class Distiller, is
now an "intermediate" loss value, instead of the final one accessible by Keras' built-in methods) and retrieving it so
that it can be used as a callback/monitoring metric in ModelCheckpoints, is not compatible with my overridden custom
version of Distiller.

Given these changes from the original, I will be testing several different versions in the hyperion: One where teacher
ModelCheckpoints use val_loss and loss as a monitoring metric and one where they use val_accuracy and accuracy, just like

The student model's CustomModelCheckpoints use val_accuracy and accuracy as a monitoring metric because
of the reasons explained above. The teacher model's ModelCheckpoints use val_loss and loss as a monitoring metric.

Hyperparameter values (alpha, train_temp, loss functions and etc) remain the same across epochs / all executions of
training loop.

After having executed this script, produced all final models and evaluated them using LoadModels.py, I edited the code
to ensure the best teacher and the best student models were re-loaded at the end of their training loops and returned by
these functions, ensuring a more object-oriented approach to training.

Key points so far: this code includes a Distiller class that is a subclass of Keras Models which overrides some of its
methods to create a wrapper class for training the student via Knowledge Distillation (KD) and a CustomModelCheckpoint
class, which is a subclass of Keras ModelCheckpoints that overrides the innit method and the set_model method to ensure
that NO DictWrapper error (as is the case when using the original Keras ModelCheckpoint) is raised, allowing the custom
class CustomModelCheckpoint to clearly access and operate on the intended Keras model
(student) being trained, regardless of it being trained within a wrapper class (Distiller).
'''

from qkeras import *
import tensorflow as tf

# Setting random seed for ensuring reproducibility of results
seed = 42
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.python.util.tf_export import keras_export

# from keras.saving import serialize_keras_object, deserialize_keras_object

print("TensorFlow version of Train_QJacobian_KD_with_Distiller.py:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, \
    Activation, Reshape, Multiply, Concatenate, Add, DepthwiseConv2D, SeparableConv2D
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
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  #keras built-in method, automatically splits
# CIFAR-10 into 50000 training images and 10000 test images. x_train = training data/the actual images, y_train = labels of
# training data. x_test = testing data/the actual images, y_test = labels of testing data.
# remember the labels are actually integer values from 0 to 9, corresponding to/representing the 10 CIFAR-10 classes (Airplane, Automobile, Bird, Cat, etc)

# TODO: TO BE REMOVED LATER!!! FOR FASTER COMPILATION, FOR TESTING IF CODE COMPILES: Keep only 10 images in the training set
x_train = x_train[:1500]
y_train = y_train[:1500]

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

# Defined to match pattern followed by original code -- even though unused in model.fit, the none gray version does exist
x_validation = x_train[:VALIDATION_SIZE]
x_train = x_train[VALIDATION_SIZE:]

# Class Distiller: wrapper class for training the student via Knowledge Distillation (KD). Class Distiller is a subclass
# of tf.keras.Model., overriding the innit(), compile(), compute_loss() and call() methods to achieve the desired functionality.
@keras_export("keras.Model", "keras.models.Model")
@keras.utils.register_keras_serializable(package="Custom")
class Distiller(tf.keras.Model):

    def __init__(self, student, teacher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha, temperature):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
            self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_pred / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        # code for easier debugging of Distiller class, i.e. making sure distillation loss was being calculated and used as intended.
        tf.print("\nstudent_loss value at this batch:", student_loss) # code suggested by chatGPT
        tf.print("distillation_loss value at this batch", distillation_loss) # code suggested by chatGPT
        tf.print("weighted average/final loss value at this batch", loss, "\n") # code suggested by chatGPT
        return loss

    def call(self, x):
        return self.student(x)

# Cosine Decay Learning\n",
import math
from keras.callbacks import Callback

class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a \"learning_rate\" attribute.')
        lr: int | float = self.eta_min + (self.eta_max - self.eta_min) * (
                    1 + math.cos(math.pi * epoch / self.T_max)) / 2
        # K.set_value(self.model.optimizer.lr, lr)
        self.model.optimizer.learning_rate.assign(lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs['lr'] = K.get_value(self.model.optimizer.lr)
        logs['lr'] = self.model.optimizer.learning_rate


#Class CustomModelCheckpoint. This is a subclass of Keras ModelCheckpoints that overrides the innit() and the set_model()
# methods to avoid DictWrapper error.
class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, model_to_use, filepath, monitor, verbose=0, save_best_only=True,
                 save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        # Initialize the base ModelCheckpoint class
        super().__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                         save_best_only=save_best_only, save_weights_only=save_weights_only,
                         mode=mode, save_freq=save_freq, options=options, **kwargs)
        self.model_to_use = model_to_use


    def set_model(self, model):
        super().set_model(self.model_to_use)


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras import optimizers
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

# epochs = 1000 is the original. epochs = 3 and epochs = 1 is only for testing whether the code can execute without errors faster
epochs = 1
# epochs = 3
# epochs = 10
# epochs = 100

# prepare data to be used in training and (it_train) iterator. Same training data instance for student and teacher models:
# it_train is passed as an argument to functions train_student() and train_teacher().
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow(x_train_gray, y_train, batch_size=64, seed=seed)  # x_train_gray, y_train values: train_data, train_labels

# learning rate policy Cosine Decay\n",
T_max = 300
lr = 1e-3
min_lr = 1e-6
cosine_decay = CosineAnnealingScheduler(T_max=epochs, eta_max=lr, eta_min=min_lr, verbose=1)

# learning rate policy ReduceLROnPlateau\n",
lrate = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, verbose=0, mode='auto',
                                          min_delta=0.0001, cooldown=0, min_lr=0)

# Keras built-in ModelCheckpoint for teacher model
# filepaths changed from original (that leading to folder ModelsKDv2) to ensure that the files containing the final results/models are not accidentally overwritten
teacher_b_v = tf.keras.callbacks.ModelCheckpoint(filepath='DuplicateResultsFromModelsKDv2/_best_vJac_teacher.keras', monitor='val_loss', save_best_only=True, mode='auto')
teacher_b_t = tf.keras.callbacks.ModelCheckpoint(filepath='DuplicateResultsFromModelsKDv2/_best_tJac_teacher.keras', monitor='loss', save_best_only=True, mode='auto')

def get_teacher():
    num_channels = 1
    image_size = 32
    num_labels = 10

    input_shape_gray = (32, 32, 1)  # default size for gray \n", # default size for cifar10
    num_classes = 10  # default class number for \n", # default class number for cifar10
    batch_norm_momentum = 0.9
    batch_norm_eps = 1e-4
    # inputs = Input(shape=input_shape)\n",
    inputs = Input(input_shape_gray)  # Input layer, whatever is appropriate for CIFAR-10

    x = QActivation("quantized_bits(8, 7, alpha=1)", name="act_0")(inputs)
    x = QConv2D(128, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_AE")(x)
    x = QActivation("quantized_relu(2)", name="act_1AE")(x)
    x = QConv2D(128, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_AD")(x)
    x = QActivation("sigmoid", name="act_1AD")(x)
    x = QConv2D(32, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_A")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    y = QDepthwiseConv2D((3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         bias_quantizer="quantized_bits(3)",
                         padding='same',
                         name="depthconv_1")(x)
    # y = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(y)
    x = Concatenate()([x, y])

    x = QActivation("quantized_relu(2)", name="act_1")(x)
    x = QConv2D(32, (1, 1),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_B")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = Add()([x, y])
    x = QActivation("quantized_relu(2)", name="act_2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = QConv2D(128, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_C")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)

    y = QDepthwiseConv2D((3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         bias_quantizer="quantized_bits(3)",
                         padding='same',
                         name="depthconv_2")(x)
    y = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(y)
    x = Concatenate()([x, y])
    x = QActivation("quantized_relu(2)", name="act_3")(x)
    x = QConv2D(128, (1, 1),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_D")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = Add()([x, y])
    x = QActivation("quantized_relu(2)", name="act_4")(x)
    x = QSeparableConv2D(32, (3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         pointwise_quantizer="quantized_bits(3, 0, 1)",
                         bias_quantizer="quantized_bits(3)",

                         name="sepconv_1")(x)
    x = QActivation("quantized_relu(2)", name="act_5")(x)
    x = tf.keras.layers.Flatten()(x)
    x = QDense(256,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense1")(x)
    x = QDense(256,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense2")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = QActivation("quantized_relu(2)", name="act_7")(x)
    x = QDense(10,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense3")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    print('Output shape x teacher', x.shape)
    print('Output shape y teacher', y.shape)

    # Instantiate model.\n",
    teacher_model = Model(inputs=inputs, outputs=x)

    return teacher_model

# call function for "getting" untrained teacher model
teacher_model = get_teacher()
def train_teacher(teacher_model, it_train):  # Standard neural network training procedure.
    # x_train_gray, y_train values: train_data, train_labels. In this function, TEACHER IS TRAINED.

    opt = tf.keras.optimizers.legacy.Adamax()
    teacher_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

    history = teacher_model.fit(it_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[cosine_decay, teacher_b_v, teacher_b_t],
                                validation_data=(x_validation_gray, y_validation))

    teacher_model_accuracy = history.history['accuracy']
    print("TRAINING ACCURACY of the TEACHER model at each epoch: ", teacher_model_accuracy)
    teacher_model_accuracy = history.history['val_accuracy']
    print("VALIDATION ACCURACY of the TEACHER model at each epoch: ", teacher_model_accuracy)
    for key, value in history.history.items():
        print("Metrics recorded during training and validation of TEACHER model at each epoch:", key, value)

    #filepaths changed from original (that leading to folder ModelsKDv2) to ensure that the files containing the final results/models are not accidentally overwritten
    teacher_model.save(filepath="DuplicateResultsFromModelsKDv2/Jac_KD_with_Distiller_teacher.keras", overwrite=True)

    reconstructed_model_teacher_b_v = tf.keras.models.load_model(
        "DuplicateResultsFromModelsKDv2/_best_vJac_teacher.keras",
        custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D,
                        'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)

    return reconstructed_model_teacher_b_v, teacher_model_accuracy # best teacher model in terms of validation accuracy is returned, but final model after training is complete is also saved, along with the best teacher model in terms of validation accuracy and best teacher model in terms of training accuracy


def get_student():
    num_channels = 1
    image_size = 32
    num_labels = 10

    input_shape_gray = (32, 32, 1)  # default size for gray \n", # default size for cifar10
    num_classes = 10  # default class number for \n", # default class number for cifar10
    batch_norm_momentum = 0.9
    batch_norm_eps = 1e-4
    # inputs = Input(shape=input_shape)\n",
    inputs = Input(input_shape_gray)  # Input layer, whatever is appropriate for CIFAR-10

    x = QActivation("quantized_bits(8, 7, alpha=1)", name="act_0")(inputs)
    x = QConv2D(96, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_AE")(x)
    x = QActivation("quantized_relu(2)", name="act_1AE")(x)
    x = QConv2D(96, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_AD")(x)
    x = QActivation("sigmoid", name="act_1AD")(x)
    x = QConv2D(24, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_A")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    y = QDepthwiseConv2D((3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         bias_quantizer="quantized_bits(3)",
                         padding='same',
                         name="depthconv_1")(x)
    # y = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(y)
    x = Concatenate()([x, y])

    x = QActivation("quantized_relu(2)", name="act_1")(x)
    x = QConv2D(24, (1, 1),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_B")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = Add()([x, y])
    x = QActivation("quantized_relu(2)", name="act_2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = QConv2D(96, (3, 3),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_C")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)

    y = QDepthwiseConv2D((3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         bias_quantizer="quantized_bits(3)",
                         padding='same',
                         name="depthconv_2")(x)
    y = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(y)
    x = Concatenate()([x, y])
    x = QActivation("quantized_relu(2)", name="act_3")(x)
    x = QConv2D(96, (1, 1),
                kernel_quantizer="stochastic_ternary",
                bias_quantizer="quantized_po2(4)",
                name="conv2d_D")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = Add()([x, y])
    x = QActivation("quantized_relu(2)", name="act_4")(x)
    x = QSeparableConv2D(24, (3, 3),
                         depthwise_quantizer="quantized_bits(4, 0, 1)",
                         pointwise_quantizer="quantized_bits(3, 0, 1)",
                         bias_quantizer="quantized_bits(3)",

                         name="sepconv_1")(x)
    x = QActivation("quantized_relu(2)", name="act_5")(x)
    x = tf.keras.layers.Flatten()(x)
    x = QDense(256,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense1")(x)
    x = QDense(256,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense2")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = QActivation("quantized_relu(2)", name="act_7")(x)
    x = QDense(10,
               kernel_quantizer="quantized_bits(3,0,1)",
               bias_quantizer="quantized_bits(3)",
               name="dense3")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, epsilon=batch_norm_eps, scale=False)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    print('Output shape x student', x.shape)
    print('Output shape y student', y.shape)

    # Instantiate model.\n",
    student_model = Model(inputs=inputs, outputs=x)

    return student_model

# call function for training teacher
teacher_trained, acct = train_teacher(teacher_model, it_train)
# call function for "getting" untrained student
student_model = get_student()
def train_student(student_model, teacher_trained, it_train): # Standard neural network training procedure.
# Same model as teacher but shortened (i.e. number of filters in QConv2D and QSeparableConv2D has been reduced).
# In this function, STUDENT IS TRAINED via the process of KD.

    distiller = Distiller(student_model, teacher_trained)

    opt = tf.keras.optimizers.legacy.Adamax()

    distiller.compile(
        optimizer=opt,
        metrics=['accuracy'],
        student_loss_fn=keras.losses.CategoricalCrossentropy(), # standard for Distiller class approach to KD. This is later used in compute_loss method of Distiller class
        distillation_loss_fn=keras.losses.KLDivergence(), # standard for Distiller class approach to KD. This is later used in compute_loss method of Distiller class.
        alpha=0.1,
        temperature=0.1 # 10 is Kera's default, 0.1 is the value that gave highest results in intermediate testing.
    )

    # filepaths changed from originals (those leading to folder ModelsKDv2) to ensure that the files containing the final results/models are not accidentally overwritten
    student_b_v = CustomModelCheckpoint(model_to_use=distiller.student,
                                filepath='DuplicateResultsFromModelsKDv2/_best_vJac_student.keras',
                                monitor='val_accuracy', save_best_only=True, mode='auto')
    student_b_t = CustomModelCheckpoint(model_to_use=distiller.student,
                                filepath='DuplicateResultsFromModelsKDv2/_best_tJac_student.keras',
                                monitor='accuracy', save_best_only=True, mode='auto')

    history = distiller.fit(it_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[cosine_decay, student_b_v, student_b_t],
                            validation_data=(x_validation_gray, y_validation))

    student_model_accuracy = history.history['accuracy']
    print("TRAINING ACCURACY of the COMPRESSED STUDENT model at each epoch: ", student_model_accuracy)
    student_model_accuracy = history.history['val_accuracy']
    print("VALIDATION ACCURACY of the COMPRESSED STUDENT model at each epoch: ", student_model_accuracy)
    for key, value in history.history.items():
        print("Metrics recorded during training and validation of COMPRESSED STUDENT model at each epoch:", key, value)
    print("student_model.summary()", student_model.summary())

    student_trained = distiller.student
    student_trained.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    student_trained.save(filepath="DuplicateResultsFromModelsKDv2/Jac_KD_with_Distiller_student.keras", overwrite=True)

    reconstructed_model_student_b_v = tf.keras.models.load_model(
        "DuplicateResultsFromModelsKDv2/_best_vJac_student.keras",
        custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D,
                        'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)

    return reconstructed_model_student_b_v # best student model in terms of validation accuracy is returned, but final model that results after training is complete
                                           # is also saved, along with the best student model in terms of validation accuracy and best student model in terms of training accuracy

# call function for training student
student_trained = train_student(student_model, teacher_trained, it_train)

K.clear_session()
print("Session cleared") # Clear memory

