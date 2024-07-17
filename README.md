# Traffic Sign Recognition and Counterexample Validation

## Summary
This repository contains scripts and models for traffic sign recognition and counterexample testing. It includes various architectures tested on datasets like GTSRB, Belgium, and Chinese traffic signs. The repository also contains tools for handling VNNLIB files, performing inference, and validating counterexamples. The tests done in output are either classification - output of the new models or verification - validation of the counterexamples obtained from VNNComp2023 from four verification tools.



## Requirements
You have to install:
- Tensorflow
```
pip install tensorflow==2.14.0
```
- Keras
```
pip install keras==2.14.0
```
- Larq
```
pip install larq==0.13.3
```


## Datasets
- [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&language=Python)
- [Belgium](https://www.kaggle.com/datasets/shazaelmorsh/trafficsigns)
- [Chinese](https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs)


In the workspace you should have following folders:
- datasets
  - Belgian_dataset
  - Chinese_dataset
  - GTSRB_dataset


## Repository Structure

### Models
  - models/onnx/: Contains different architectures of models used for the tests:
      - 3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx (size 30)
      - 3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx (size 48)
      - 3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx (size 64)
These models were used in VNNCOMP2023. In the verification tasks these models were used, see output/verification/.

### Properties
  - properties/vnnlib/: Contains all the VNNLIB 2023 files used for the tests.

### Source Code
  - src/main.py: Main script for running the classifier (3QConv) and handling datasets.
  - src/classifiers: folders which contain different classifiers for each dataset. It contains additional architectures compared to main.py, e.g. 2QConv_german.py, 3QConv_german_v2.py and xnor.py.
  - src/utils/confusion_matrix.py: Plots the confusion matrix for a model.
  - src/utils/counterexample_to_png.py: Converts the counterexample file from SMT-LIB format to PNG for quality checking.
  - src/utils/find_max_valY.py: Checks for the class predicted by the verification tool in the counterexample (the maximum value of Y).
  - src/utils/inference_from_cex.py: Performs inference directly from counterexamples.
  - src/utils/inference_from_png.py: Performs inference on counterexamples transformed into PNG.
  - src/utils/smt2.py: Script to check the validity of counterexamples.

### Output Folder
  - output/classification contains results (confusion_matrix, training_plots, training_summary) obtained from the classifiers in src/classifiers for each of the three datasets
  - output/verification:
    - /counterexamples contains the counterexamples from VNNCOMP2023 for each of the four solvers
    - /img_from_counterexamples the PNG images obtained after trasforming the counterexamples above
    - /testing_counterexample_quality results obtained by checking if the counterexample obtained in the VNNCOMP 2023 satisfies the coresponding VNN-LIB file.


## How to run the training scripts
Depending on which architecture you want to train, you can choose to run main.py if you want a 3QConv architecture or go to src/classifiers if you want to run the classifiers separately and have more architectures.

Before runing the scripts, make sure you have following folders created in your workspace:
- datasets
  - Belgian_dataset
  - Chinese_dataset2
  - GTSRB_dataset
- output
  - German (do the same for Belgian and Chinese)
    - confusion_matrix
      - 30x30
      - 48x48
      - 64x64
    - training_plots
      - 30x30
      - 48x48
      - 64x64
    - training_summary
      - 30x30
      - 48x48
      - 64x64


At the end of each script from src/classifiers there is the `main` code which calls the functions for training. If you don't want to train all variations of architectures, you can comment out a certain training.


## The tests
[Folder](https://drive.google.com/drive/folders/1V1hoi4S70QxZqYTWhEB-kXo_ZojoHu6i?usp=drive_link) contains the following verification results:
- [File](https://docs.google.com/spreadsheets/d/1Xd-27N0P-cWXvk6QhlAgW6nDSljaf5fLwNIL7iop8b4/edit?usp=drive_link) checks if the counterexample given by the verification tool is indeed a valid counterexample. A counterexample is valid if its class as inferred by the BNN model is different than the class of the initial perturbed image. The results are for the counterexamples from the VNN-COMP2023 as well as for newer versions of the tools see paper from [here](https://drive.google.com/drive/folders/1V1hoi4S70QxZqYTWhEB-kXo_ZojoHu6i?usp=drive_link).
- [File](https://docs.google.com/spreadsheets/d/1Xd-27N0P-cWXvk6QhlAgW6nDSljaf5fLwNIL7iop8b4/edit?usp=drive_link) checks if the counterexample given by the verification tool is within the bounds of the corresponding VNN-LIB2023 file. The results are for the counterexamples from the VNN-COMP2023 as well as for newer versions of the tools see paper from [here](https://drive.google.com/file/d/1G-dkY5EIA4_xF-PgPAfqNt9-gNgedf01/view?usp=drive_link).
