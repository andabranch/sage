from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from keras.utils import to_categorical

def get_testing_dataset_chinese():
    data = []
    labels = []
    CHINESE_DATASET_PATH = 'datasets/Chinese_dataset2'

    annotations_file = os.path.join(CHINESE_DATASET_PATH, 'TestAnnotation.txt')
    annotations_df = pd.read_csv(annotations_file, delimiter=';')

    for index, row in annotations_df.iterrows():
        try:
            image_path = os.path.join(
                CHINESE_DATASET_PATH, 'Test', row['file_name'])
            image = Image.open(image_path)
            image = image.resize(IMG_RESIZE)
            image = np.array(image)
            data.append(image)
            labels.append(row['category'])
        except Exception as e:
            print(f"Error loading image: {e}")

    data = np.array(data)
    labels = np.array(labels)

    labels_categorical = to_categorical(labels, CLASSES)

    print(annotations_df[annotations_df['file_name'].apply(lambda x: not isinstance(x, str))])

    print(data.shape, labels_categorical.shape)
    return data, labels_categorical

def confusion_matrix_chinese(model, X_test, y_test, class_names, save_path):
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cr = classification_report(y_true_labels, y_pred_labels, target_names=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()

    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(cr)

    print(cr)

confusion_matrix_chinese()