import importlib
import os
from utils.images_per_class import *

def choose_dataset():
    print("Choose dataset: Chinese, Belgian, German")
    dataset = input().lower()
    return dataset

def choose_neurons():
    print("Choose the number of neurons for the last dense layer (0, 64, 128, 256, 512, 1024): ")
    neurons = int(input())
    return neurons

def images_per_class(dataset):
    dataset_module = importlib.import_module("utils.images_per_class")

    if dataset == "chinese":
        return dataset_module.images_per_class_chinese(dataset_path='datasets/Chinese_dataset2')
    elif dataset == "belgian":
        return dataset_module.images_per_class_belgian(dataset_path='datasets/Belgian_dataset')
    elif dataset == "german":
        return dataset_module.images_per_class_german(dataset_path='datasets/GTSRB_dataset')
    else:
        print("Invalid dataset.")
        return None, None


def class_count():
    while True:
        dataset = choose_dataset()
        if dataset is None:
            print("Invalid dataset. Please try again.")
            continue

        train_class_counts, test_class_counts = images_per_class(dataset)

        if train_class_counts is not None and test_class_counts is not None:
            print("Training Dataset Class Counts:")
            print(train_class_counts)

            print("Testing Dataset Class Counts:")
            print(test_class_counts)

            save_class_counts(dataset, train_class_counts, test_class_counts)
        else:
            print(f"Error counting classes for {dataset}. Please check your implementation.")
            continue

        answer = input("Check for another dataset? (Yes/No): ").lower()
        if answer != "yes":
            break
###
def save_class_counts(dataset, train_class_counts, test_class_counts):
    filename = f"{dataset}_classes.txt"
    
    try:
        with open(filename, "w") as file:
            file.write("Training Dataset Class Counts:\n")
            for class_name, count in train_class_counts.items():
                file.write(f"{class_name}: {count} images\n")
            
            file.write("\nTesting Dataset Class Counts:\n")
            for class_name, count in test_class_counts.items():
                file.write(f"{class_name}: {count} images\n")

        print(f"Class counts saved to {filename} successfully.")
    except Exception as e:
        print(f"Error saving class counts: {e}")
        raise
    
def run_architecture():
    dataset = choose_dataset()
    print("I will do the splitting now..")

    dataset_module = importlib.import_module(f"3QConv_{dataset}")

    data, labels = dataset_module.get_training_dataset()
    X_train, X_validation, y_train, y_validation = dataset_module.split_in_training_n_test(data, labels)

    X_test, y_test = dataset_module.get_testing_dataset()

    neurons = choose_neurons()

    if neurons == 0:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    elif neurons == 64:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_64_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    elif neurons == 128:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_128_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    elif neurons == 256:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_256_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    elif neurons == 512:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_512_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    elif neurons == 1024:
        dataset_module.QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43(
            X_train, X_validation, y_train, y_validation, X_test, y_test)
    else:
        print("Invalid number of neurons. Please choose from (0, 64, 128, 256, 512, 1024).")
        
        
if __name__ == "__main__":
    #class_count()
    run_architecture()

    
