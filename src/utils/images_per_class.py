import os
import pandas as pd

def images_per_class_german(dataset_path):
    train_df = pd.read_csv(os.path.join(dataset_path, 'Train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'Test.csv'))

    train_class_counts = train_df['ClassId'].value_counts().sort_index()
    test_class_counts = test_df['ClassId'].value_counts().sort_index()

    return train_class_counts, test_class_counts



def images_per_class_chinese(dataset_path):
    train_annotation_file = os.path.join(dataset_path, 'TrainAnnotation.txt')
    train_df = pd.read_csv(train_annotation_file, delimiter=';')

    train_class_counts = train_df['category'].value_counts().sort_index()

    test_annotation_file = os.path.join(dataset_path, 'TestAnnotation.txt')
    test_df = pd.read_csv(test_annotation_file, delimiter=';')

    test_class_counts = test_df['category'].value_counts().sort_index()

    return train_class_counts, test_class_counts



def images_per_class_belgian(dataset_path):
    train_df = pd.read_csv(os.path.join(dataset_path, 'Train.csv'), sep=';')
    test_df = pd.read_csv(os.path.join(dataset_path, 'Test.csv'), sep=';')

    train_class_counts = train_df['ClassId'].value_counts().sort_index()
    test_class_counts = test_df['ClassId'].value_counts().sort_index()

    return train_class_counts, test_class_counts

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
