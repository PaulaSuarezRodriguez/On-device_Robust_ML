'''
This script, Attack_KD_with_Distiller_models.py, takes inspiration from the original script
"Attacks.py" (from QKerasCodes) provided to me by my supervisor. I have heavily edited the original code, as well as
added new functionality. For further details, refer to the relevant section of my Final Report.
'''
from tabnanny import verbose

# ORIGINAL Attack.py IMPORTS:
import numpy as np
from numpy import save
import tensorflow
import os
# import cv2 # import commented out because only usage of cv2 was commented out in original code
from keras import models
import keras
from keras.models import save_model
from qkeras import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation, Reshape, Multiply, Concatenate, Add
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import backend as K
import tempfile
import tensorflow_model_optimization as tfmot
from keras.callbacks import LearningRateScheduler
#from setup_cifar import CIFAR
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from PIL import Image
import os

# Setting random seed for ensuring reproducibility of results
seed = 42
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.python.keras.backend import epsilon

from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from tensorflow.python.framework.ops import disable_eager_execution
from art.attacks.evasion import CarliniL0Method, AutoAttack
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
#from art.attacks.evasion import LaserAttack # too complicated, requires extra objects
#from art.attacks.evasion import AutoAttack # is like the loop I am implementing -- it implements several attacks
#from art.attacks.evasion import ShadowAttack #ValueError: This attack only accepts a single sample as input. No batches ergo, useless to me
from art.attacks.evasion import SquareAttack
#from art.attacks.evasion import pe_malware_attack # not useful
#from art.attacks.evasion import adversarial_asr # not useful
from art.attacks.evasion import AdversarialPatch # needs extra library tensorflow_addons. Installed without issues. Currently being used. Monitor
#from art.attacks.evasion import adversarial_texture # not callable
#from art.attacks.evasion import AutoConjugateGradient #ValueError: AutoProjectedGradientDescent is expecting logits as estimator output, the provided estimator seems to predict probabilities.
#from art.attacks.evasion import AutoProjectedGradientDescent #ValueError: AutoProjectedGradientDescent is expecting logits as estimator output, the provided estimator seems to predict probabilities.
from art.attacks.evasion import BoundaryAttack
#from art.attacks.evasion.brendel_bethge import BrendelBethgeAttack #NOT USED. needs extra library numba. Installed without issues.import Currently being used. HOWEVEEEER: MonitorValueError: The provided estimator seems to predict probabilities. If loss_type='difference_logits_ratio' the estimator has to to predict logits.
#from art.attacks.evasion.composite_adversarial_attack import CompositeAdversarialAttackPyTorch # discarded. Only for Pytorch models. Discarded!!!
#from art.attacks.evasion import DecisionTreeAttack #not for images, discarded!!!
from art.attacks.evasion import DeepFool
#from art.attacks.evasion import DPatch #discarded, not for TensorFlowV2Classifier
#from art.attacks.evasion.dpatch_robust import RobustDPatch #discarded, not for TensorFlowV2Classifier
from art.attacks.evasion import ElasticNet
#from art.attacks.evasion import FeatureAdversariesTensorFlowV2 #discard, incompatible, ValueError: The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.
#from art.attacks.evasion import FrameSaliencyAttack #too complex, requires additional evasion attack, discarded.
from art.attacks.evasion import GeoDA
#from art.attacks.evasion import GRAPHITEBlackbox #discarded, incompatible. ValueError: Target labels `y` need to be provided.
# from art.attacks.evasion import GRAPHITEWhiteboxPyTorch #discarded, not for TensorFlowV2Classifier
from art.attacks.evasion import HighConfidenceLowUncertainty #discarded, not for TensorFlowV2Classifier
from art.attacks.evasion import HopSkipJump
#from art.attacks.evasion import ImperceptibleASRPyTorch #for speech recognition models
from art.attacks.evasion.iterative_method import BasicIterativeMethod # check if it works, since it is not attack but class of attacks
#from art.attacks.evasion import LowProFool #discarded, incompatible, ValueError: No feature importance vector has been provided yet.
#from art.attacks.evasion import MomentumIterativeMethod #too complex / advanced, discarded
from art.attacks.evasion import NewtonFool
#from art.attacks.evasion import OverTheAirFlickeringPyTorch #discarded, not for TensorFlowV2Classifier #Plus, attack for video recognition models, import think.
#from art.attacks.evasion import OverloadPyTorch #discarded, not for TensorFlowV2Classifier
from art.attacks.evasion import PixelAttack
#from art.attacks.evasion import RescalingAutoConjugateGradient #discarded, I could not even find its documentation in https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#, I do not know what it does
from art.attacks.evasion import ThresholdAttack
from art.attacks.evasion.saliency_map import SaliencyMapMethod
#from art.attacks.evasion.shapeshifter import ShapeShifter #discarded, not for TensorFlowV2Classifier
from art.attacks.evasion.sign_opt import SignOPTAttack
from art.attacks.evasion.simba import SimBA
#from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation #discarded, incompatible, ValueError: Labels `y` cannot be None.
from art.attacks.evasion import UniversalPerturbation
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
#from art.attacks.evasion.wasserstein import Wasserstein #too complex / advanced, discarded.import Although, not impossible. Revisit later on
from art.attacks.evasion.zoo import ZooAttack

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, \
    classification_report
from matplotlib import pyplot as plt
import seaborn as sns

print("TensorFlow version of Attack_KD_with_Distiller_models.py:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# CODE FOR PREPROCESSING DATASET: heavily inspired by my supervisor's original code
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

VALIDATION_SIZE = 1000
# To be used in model.fit, since the original code used x_test_gray, NOT x_test
x_validation_gray = x_train_gray[:VALIDATION_SIZE]
y_validation = y_train[:VALIDATION_SIZE]
x_train_gray = x_train_gray[VALIDATION_SIZE:]
y_train = y_train[VALIDATION_SIZE:]
x_validation = x_train[:VALIDATION_SIZE]
x_train = x_train[VALIDATION_SIZE:]

#check model structure and the number of parameters\n",
train_temp = 20 #ORIGINAL; value decided upon by supervisor.
#train_temp = 0.1

# Loss function for soft results
def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted / train_temp)

# For testing the robustness of other models and achieve some of the comparative results I included in my Final Report, the code must be modified slightly. This version of the script is that for evaluating the final project results, not any intermediate results.

# DEFINITE CODE to be used
# For testing robustness of best models to be deployed on STM MCU
reconstructed_model6 = tf.keras.models.load_model("ModelsKDv2/_best_vJac_student.keras", custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D, 'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
reconstructed_teacher = tf.keras.models.load_model("ModelsKDv2/_best_vJac_teacher.keras",custom_objects={'QActivation': QActivation, 'QConv2D': QConv2D, 'QDepthwiseConv2D': QDepthwiseConv2D,'QSeparableConv2D': QSeparableConv2D, 'QDense': QDense}, compile=True)
models = {"_best_vJac_student": reconstructed_model6, "reconstructed_teacher": reconstructed_teacher} # "_best_vJac_student" is assigned name "reconstructed_model6" to avoid confusion, as in script LoadModels.py, "reconstructed_model6" is the variable name for re-loaded "_best_vJac_student"

# IMPORTANT! The attacks in "adversarial_accuracies_student_original" and "adversarial_accuracies_teacher_original" had to be executed by my supervisor on her machine, which is why the accuracies of the teacher and student models under these attacks have been hardcoded -- so that I can still be able to produce plots for comparative analysis in the Final Report's Results chapter.
adversarial_accuracies_student_original = {"DeepFool": 13.0, "GeoDA": 6.0, "HopSkipJump": 25.0, "SimBA": 75.0}  # hard_coded_attacks_student
adversarial_accuracies_teacher_original = {"DeepFool": 8.0, "GeoDA": 3.0, "HopSkipJump": 5.0, "SimBA": 73.0}  # hard_coded_attacks_teacher

def robustness(models, adversarial_accuracies_student_original, adversarial_accuracies_teacher_original):
    # build model
    for model_name, model in models.items():

        classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32, 32, 1), loss_object=fn,clip_values=[-128, 128])

        # DEFINITE CODE to be used
        #Attacks
        attacks = {#White-box attacks:
                    #"VirtualAdversarialMethod":VirtualAdversarialMethod(classifier,verbose=False),
                    #"CarliniL0Method": CarliniL0Method(classifier, targeted=False, batch_size = 32),
                    #"CarliniL2Method": CarliniL2Method(classifier, targeted=False, batch_size=32),
                   "ProjectedGradientDescent":ProjectedGradientDescent(classifier, eps = 0.3, eps_step = 0.1, max_iter = 100, targeted=False, num_random_init = 0, batch_size = 32, random_eps = False, summary_writer = False, verbose=False),
                   "FastGradientMethod":FastGradientMethod(classifier, eps = 0.3, eps_step = 0.1, targeted=False, num_random_init = 0, batch_size = 32, summary_writer = False),
                   #"AdversarialPatch":AdversarialPatch(classifier, targeted=False),
                   #"ElasticNet":ElasticNet(classifier, verbose=False),
                   "BasicIterativeMethod":BasicIterativeMethod(classifier, verbose=False),
                   #"NewtonFool":NewtonFool(classifier, verbose=False),
                   #"SaliencyMapMethod":SaliencyMapMethod(classifier, verbose=False),
                   #"UniversalPerturbation":UniversalPerturbation(classifier,attacker='deepfool',delta=0.000001,max_iter=10,eps=0.5,norm=2, verbose=False),
                    #"DeepFool": DeepFool(classifier, verbose=False), #teacher 8, student 13


                   #Black-box attacks:
                   "SquareAttack": SquareAttack(classifier, norm= np.inf, adv_criterion= None, loss= None, max_iter=100,eps=0.3,p_init=0.8,nb_restarts=1, verbose = False),
                   #"BoundaryAttack": BoundaryAttack(classifier, targeted=False, verbose = False),
                   #"GeoDA":GeoDA(classifier,verbose=False), # teacher 3, student 6
                   #"HopSkipJump":HopSkipJump(classifier, verbose=False), # teacher 5, student 25
                   #"PixelAttack":PixelAttack(classifier, targeted=False, verbose=False),
                   #"ThresholdAttack":ThresholdAttack(classifier, targeted=False, verbose=False),
                   #"SignOPTAttack":SignOPTAttack(classifier, targeted=False, verbose=False),
                   #"SimBA":SimBA(classifier, targeted=False, verbose=False), #teacher 73, student 75
                   #"ZooAttack":ZooAttack(classifier, targeted=False, verbose=False)
                   }


    # generate noise
        for attack_name, attack in attacks.items():
            print(f"Model: {model_name}, Attack: {attack_name}:")
            x_test_adv = attack.generate(x_test_gray[0:100])

            suc = 0
    # script_path = "cifarAE/SQ"
            for i in range(len(x_test_gray[0:100])):
                a = classifier.predict(x_test_adv[i:i + 1])
                a = np.argmax(a)
                b = np.argmax(y_test[i]) #new, changed position
                c = abs(a - b) #new, changed position
                if (c == False): #new, changed position
                    suc = suc + 1  # Success if actual and predicted output are same, i.e., c=0  #new, changed position
                    print('Predicted output:', a, '||| Actual output:', b, '||| Difference:', c, '=> Success (actual and predicted output are the same). Accuracy achieved so far:', suc) #new, changed position for printing b, c and suc
                else: print('Predicted output:', a, '||| Actual output:', b, '||| Difference:', c)

            adversarial_accuracy = suc*100/len(x_test_adv)
            print(f"Accuracy of {model_name} under attack {attack_name}: {adversarial_accuracy}")


            if model_name == "_best_vJac_student":
                model.compile(metrics=['accuracy'])  # in the case of student's b_v and b_t since they are CUSTOM ModelCheckpoints
                clean_accuracy = model.evaluate(x_test_gray, y_test)[1] * 100
                adversarial_accuracies_student_original.update({attack_name: adversarial_accuracy})
                print("adversarial_accuracies_student dictionary being updated:", adversarial_accuracies_student_original, "\n")

            if model_name == "reconstructed_teacher":
                adversarial_accuracies_teacher_original.update({attack_name: adversarial_accuracy})
                print("adversarial_accuracies_teacher dictionary being updated:", adversarial_accuracies_teacher_original, "\n")

    return clean_accuracy, adversarial_accuracies_student_original, adversarial_accuracies_teacher_original


# Robustness comparison functions / functions for measuring robustness
def adversarial_accuracy_fn(clean_accuracy, adversarial_accuracies_student):
    for attack_name, adversarial_accuracy in adversarial_accuracies_student.items():
        bars = ['When predicting on\nclean accuracy', f"When predicting under attack\n{attack_name}"]
        accuracies = [clean_accuracy, adversarial_accuracies_student.get(attack_name)]
        plt.figure(figsize=(10, 6))
        plt.bar(bars, accuracies) #https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
        plt.title(f"Robustness of Best Distilled Student Model\nunder {attack_name}", fontsize=20)
        plt.xlabel('\nBest Distilled Student Model', fontsize=20)
        plt.ylabel('Test Accuracy', fontsize=20)
        plt.ylim(0, 85)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        print(f"Model: Best Distilled Student Model, Attack: {attack_name}:")
        print(f"Title of Bar Graph: Robustness of Best Distilled Student Model under {attack_name}")
        print(f"Clean Test Accuracy of Best Distilled Student Model: {clean_accuracy} %")
        print(f"Adversarial Test Accuracy of Best Distilled Student Model under {attack_name}: {adversarial_accuracy} %")
        print(f"Difference between Clean Accuracy and Adversarial Accuracy under {attack_name}: {clean_accuracy - adversarial_accuracy} %\n")

def robustness_teacher_student_fn(adversarial_accuracies_teacher, adversarial_accuracies_student):
    for attack_name, adversarial_accuracy in adversarial_accuracies_student.items():
        bars = ['Teacher model', 'Student model']
        accuracies = [adversarial_accuracies_teacher.get(attack_name), adversarial_accuracies_student.get(attack_name)]
        plt.figure(figsize=(10, 6))
        plt.bar(bars, accuracies) #https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
        plt.title(f"Increase in Robustness achieved by KD:\nRobustness of Teacher vs Student under {attack_name}", fontsize=20)
        plt.xlabel(f"\nKD models under {attack_name}", fontsize=20)
        plt.ylabel('Adversarial Test Accuracy', fontsize=20)
        plt.ylim(0, 90)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.show()
        print(f"Models: Best Distilled Student Model and Teacher Model, Attack: {attack_name}:")
        print(f"Title of Bar Graph: Increase in Robustness achieved by KD: Robustness of Teacher vs Student when under {attack_name}")
        print(f"Adversarial Test Accuracy of Teacher Model under {attack_name}: {adversarial_accuracies_teacher.get(attack_name)} %")
        print(f"Adversarial Test Accuracy of Best Distilled Student Model under {attack_name}: {adversarial_accuracies_student.get(attack_name)} %")
        print(f"Difference in Accuracy between Best Teacher and Best Distilled Student under {attack_name}: {adversarial_accuracies_teacher.get(attack_name) - adversarial_accuracies_student.get(attack_name)} %\n")


if __name__ == "__main__":
    clean_accuracy, adversarial_accuracies_student, adversarial_accuracies_teacher = robustness(models, adversarial_accuracies_student_original, adversarial_accuracies_teacher_original)
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)
    adversarial_accuracy_fn(clean_accuracy, adversarial_accuracies_student)
    print("\n" * 3, "/" * 200, "\n", "/" * 200, "\n", "/" * 200, "\n" * 3)
    robustness_teacher_student_fn(adversarial_accuracies_teacher, adversarial_accuracies_student)


K.clear_session()
print("Session cleared") # Clear memory

