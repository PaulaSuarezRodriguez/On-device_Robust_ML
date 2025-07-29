# IndividualProject_220029768-main
Name: Paula Suarez Rodriguez

Student number: 220029768

Student Email: Paula.Suarez-Rodriguez@city.ac.uk

Github Repository (currently only available to supervisor): https://github.com/PaulaSuarezRodriguez/IndividualProject_220029768.git 

Individual Project "On-Device and Online Machine Learning for Defence against Adversarial Attacks"



## Understanding the code:
This project is composed of three main scripts:
- Train_QJacobian_KD_with_Distiller.py
- LoadModels.py
- Attack_KD_with_Distiller_models.py

Train_QJacobian_KD_with_Distiller.py produces the machine learning models that are the main output of this project. 

These machine learning models are stored in .keras files. For evaluation or further use of these models, they need to be re-loaded into the project. LoadModels.py re-loads the models from memory and evaluates them to determine the success of the first objective of the project.

For the second objective, the models are again re-loaded by script Attack_KD_with_Distiller_models.py. This script evaluates them to determine the success of the second objective of the project.

## Dataset:
The dataset used for training, validating and testing machine learning models in this project is CIFAR-10. The dataset is directly loaded from the Tensorflow/Keras library. There is no need to download the library. As long as the environment for executing this project is set up correctly, the dataset will be made accessible automatically, with no additional set-up required.

## Requirements:
Essential libraries: python=3.10.13, tensorflow=2.12.0 and qkeras==0.9.0

The environment required to run this project has been exported and added to the submission as a .yaml file and as a .txt file. These files are “QkerasCodes_updated_python3.10_PyTorch_matplotlib_conda_env.yml” and “QkerasCodes_updated_python3.10_PyTorch_matplotlib_conda_env.txt”. They both contain the same information; I have provided the environment requirements in two different file formats only for the convenience of the marker. 

However, the essential requirements to execute this project are: python=3.10.13, tensorflow=2.12.0 and qkeras==0.9.0. It is essential that the adequate versions of these libraries and their dependancies are installed. Other versions of python or tensorflow may be incompatible with qkeras==0.9.0, particularly if executing on a MacBook.


## How to run the code to duplicate results: 
In summary, to execute the code of this project: 

- Run script Train_QJacobian_KD_with_Distiller.py. This script produces the machine learning models that are the final outcome of this project 
- Run script LoadModels.py. This script evaluates the machine learning models produced by Train_QJacobian_KD_with_Distiller.py.
- Run script Attack_KD_with_Distiller_models.py. This script evaluates the machine learning models produced by Train_QJacobian_KD_with_Distiller.py.

The results from Train_QJacobian_KD_with_Distiller.py will be stored in folder “DuplicateResultsFromModelsKDv2” to avoid overriding the final results of the project, which are located in “ModelsKDv2”. 

The outcome from executing scripts LoadModels.py and Attack_KD_with_Distiller_models.py can be observed in the console (they do not produce external files). By default, LoadModels.py and Attack_KD_with_Distiller_models.py will re-load and evaluate the .keras files in folder “ModelsKDv2”, I.e. the machine learning models that are the final outcome of this project. If you wish to change this behaviour, read the extended explanation below:

In order to duplicate the results of this project, the first script to execute is Train_QJacobian_KD_with_Distiller.py. The final results/models produced from this script that are then re-loaded and used in LoadModels.py and Attack_KD_with_Distiller_models.py were stored during execution in folder “ModelsKDv2”. In order to avoid these results from being accidentally overwritten when executing Train_QJacobian_KD_with_Distiller.py to duplicate the results achieved by this project, I have changed the file paths in Train_QJacobian_KD_with_Distiller.py to a different folder named “DuplicateResultsFromModelsKDv2”. 

Additionally, I have tried to make the file paths independent of any local directories (I.e. relative file paths), as the original file paths were specific to my local machine. If when executing any one of the scripts, these fail due to a “directory not found” error, change the file paths to match the location of my project folder on your machine; e.g. change “filepath="DuplicateResultsFromModelsKDv2/Jac_KD_with_Distiller_student.keras”” to “filepath="path/to/my/project/DuplicateResultsFromModelsKDv2/Jac_KD_with_Distiller_student.keras””, replacing "path/to/my/project" with the actual location of the project folder on your system, effectively providing the absolute path name.

In scripts LoadModels.py and Attack_KD_with_Distiller_models.py, the models being re-loaded by default are those from folder “ModelsKDv2”, as these are the final results of the project. Thus, in executing these files, the console output will show the results presented in the Results section of the final report. 
If you wish to re-execute Train_QJacobian_KD_with_Distiller.py to produce a new set of results, these results will be stored in folder “DuplicateResultsFromModelsKDv2”. Thus, in order to re-load and evaluate these new set of results to verify the entire project code works as intended, the file paths in scripts LoadModels.py and Attack_KD_with_Distiller_models.py, will need to be changed to access the models in “DuplicateResultsFromModelsKDv2” instead of default “ModelsKDv2”; e.g. change “filepath="DuplicateResultsFromModelsKDv2/Jac_KD_with_Distiller_student.keras”” to “filepath="ModelsKDv2/Jac_KD_with_Distiller_student.keras”.

WARNINGS:
- Do NOT overwrite “ModelsKDv2” folder! If you must change the relative file paths provided in script Train_QJacobian_KD_with_Distiller.py to absolute file paths, ensure the final destination of the files remains folder “DuplicateResultsFromModelsKDv2”, and not “ModelsKDv2”. 
- Do NOT changes the names of the .keras files! If you wish to evaluate a new set of results or you need to change the relative file paths to absolute file paths, ensure the name of the .keras file remains unchanged or the outcomes of the entire code will be disrupted. 


## Where (aside from Moodle submission area) can the code be found: 
Github URL: https://github.com/PaulaSuarezRodriguez/IndividualProject_220029768.git

This GitHub repository is private and only myself and my supervisor (Dr. Ferheen Ayaz) have access to it, as requested by the module teaching team. 

In this GitHub repository, in addition to the final version of my code uploaded to the Moodle submission area for marking, all intermediate  versions can also be found. 


## Additional information: 
Qkeras models such as the ones generated by script Train_QJacobian_KD_with_Distiller.py require training for 1000 epochs to achieve the performance presented in the results section of the report. This will take approximately 8-10h to execute on GPU. 

In this folder, you will also find a copy of the plots produced by LoadModels.py and Attack_KD_with_Distiller_models.py. These are NOT automatically saved when executing the code, as is the case with the .keras files. I have manually saved the plots produced from executing LoadModels.py and Attack_KD_with_Distiller_models.py when evaluating the final results/models of this project for refecence.
