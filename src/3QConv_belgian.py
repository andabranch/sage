from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
import contextlib
import larq as lq
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# CONSTANTS
CLASSES = 62
SIZE = 30
IMG_RESIZE = (SIZE, SIZE)
CURRENT_PATH = os.getcwd()
OUTPUT_PATH = 'output/Belgian/'
FILTER_128 = 128
FILTER_64 = 64
FILTER_32 = 32
FILTER_16 = 16
KERNEL_SIZE_5 = (5, 5)
KERNEL_SIZE_3 = (3, 3)
KERNEL_SIZE_2 = (2, 2)
NO_DENSE = 0
DENSE_64 = 64
DENSE_128 = 128
DENSE_256 = 256
DENSE_512 = 512
DENSE_1024 = 1024

USE_BN = True
USE_MP = True
USE_MP3 = True
NO_BN = False
NO_MP = False

BATCH_SIZE = 32
EPOCHS = 30
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")


def get_training_dataset():
    data = []
    labels = []
    BELGIAN_DATASET_PATH = 'datasets/Belgian_dataset/BelgiumTSC_Training'

    for i in range(CLASSES):
        class_id = f'{i:05d}'
        path = os.path.join(BELGIAN_DATASET_PATH, 'Training', class_id)
        images = [f for f in os.listdir(path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.gif'))]

        for img in images:
            try:
                image_path = os.path.join(path, img)
                image = Image.open(image_path)
                image = image.resize(IMG_RESIZE)
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image: {e}")

    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    return data, labels


def get_testing_dataset():
    data = []
    labels = []
    BELGIAN_DATASET_PATH = 'datasets/Belgian_dataset/BelgiumTSC_Testing'

    for i in range(CLASSES):
        class_id = f'{i:05d}'
        path = os.path.join(BELGIAN_DATASET_PATH, 'Testing', class_id)
        images = [f for f in os.listdir(path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.gif'))]

        for img in images:
            try:
                image_path = os.path.join(path, img)
                image = Image.open(image_path)
                image = image.resize(IMG_RESIZE)
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image: {e}")

    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    return data, labels


def split_in_training_n_test(data, labels):
    X_train, X_validation, y_train, y_validation = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)
    y_train = to_categorical(y_train, CLASSES)
    y_validation = to_categorical(y_validation, CLASSES)
    return X_train, X_validation, y_train, y_validation


def build_model(filter1, kernel_size1, filter2, kernel_size2, filter3, kernel_size3, neurons_dense1, use_batchnormalization, use_maxpooling):
    # Building the model
    model = tf.keras.models.Sequential()

    # In the first layer we only quantize the weights and not the input
    model.add(lq.layers.QuantConv2D(filter1, kernel_size1,
                                    kernel_quantizer="ste_sign",
                                    kernel_constraint="weight_clip",
                                    use_bias=False,
                                    input_shape=(SIZE, SIZE, 3)))
    if use_maxpooling:
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if use_batchnormalization:
        model.add(tf.keras.layers.BatchNormalization(scale=False))

    # Block 2
    model.add(lq.layers.QuantConv2D(
        filter2, kernel_size2, use_bias=False, **kwargs))
    if use_maxpooling:
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if use_batchnormalization:
        model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(
        filter3, kernel_size3, use_bias=False, **kwargs))

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(tf.keras.layers.Flatten())

    if neurons_dense1 != 0:
        model.add(lq.layers.QuantDense(
            neurons_dense1, use_bias=False, **kwargs))
        if use_batchnormalization:
            model.add(tf.keras.layers.BatchNormalization(scale=False))

    # Output layer
    model.add(lq.layers.QuantDense(CLASSES, use_bias=False, **kwargs))
    model.add(tf.keras.layers.Activation("softmax"))

    return model


def compile_n_fit(model, X_train, X_validation, y_train, y_validation):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, validation_data=(X_validation, y_validation))
    return history


def save_summary(model, test_name):
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            lq.models.summary(model)


def plot_graphs(history, test_name):
    df = pd.DataFrame(history.history).rename_axis(
        'epoch').reset_index().melt(id_vars=['epoch'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, mtr in zip(axes.flat, ['loss', 'accuracy']):
        ax.set_title(f'{mtr.title()} Plot')
        dfTmp = df[df['variable'].str.contains(mtr)]
        sns.lineplot(data=dfTmp, x='epoch', y='value', hue='variable', ax=ax)
    fig.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_PATH + 'training_plots/' + test_name + '.png')



def get_n_save_test_metrics(model, X_test, y_test, test_name):
    pred = model.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, np.argmax(pred, axis=1), average='weighted')
    
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\nOther Test Metrics:')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')

    print('\nTest Metrics:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


def get_n_save_test_accuracy(model, X_test, y_test, test_name):
    pred = model.predict(X_test)
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\nTest Accuracy:')
            print(accuracy_score(y_test, np.argmax(pred, axis=1)))
            
    print(accuracy_score(y_test, np.argmax(pred, axis=1)))


def get_n_save_execution_time(start, end, test_name):
    elapsed = (end - start).total_seconds() / 60
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\nTest Execution time: ' + str(elapsed) + ' minutes.')


def generate_and_save_confusion_matrix(model, X_test, y_test, test_name, dataset_name):
    y_pred = model.predict(X_test)

    y_pred_labels = np.argmax(y_pred, axis=1)

    if y_test.ndim > 1:
        y_true_labels = np.argmax(y_test, axis=1)
    else:
        y_true_labels = y_test

    cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

    unique_classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)

    fig, ax = plt.subplots(figsize=(30, 30))
    cmap = plt.cm.Blues
    disp.plot(cmap=cmap, ax=ax, values_format='.2f')
    plt.xticks(rotation=90)
    plt.yticks(range(len(unique_classes)), unique_classes)
    plt.title(f'{dataset_name} Dataset - {test_name}')
    plt.savefig(OUTPUT_PATH + 'confusion_matrix/' + f'{test_name}_{dataset_name}.png')



def post_build_process(model, X_train, X_validation, y_train, y_validation, X_test, y_test, test_name):
    history = compile_n_fit(
        model, X_train, X_validation, y_train, y_validation)

    model.save(OUTPUT_PATH + 'models/' + test_name + '.h5')

    save_summary(model, test_name)
    plot_graphs(history, test_name)

    dataset_name = "Belgian"

    generate_and_save_confusion_matrix(model, X_test, y_test, test_name, dataset_name)
    
    get_n_save_test_accuracy(model, X_test, y_test, test_name)

    get_n_save_test_metrics(model, X_test, y_test, test_name)



def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, NO_DENSE, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)


def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_256_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_256_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, DENSE_256, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)


def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_128_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_128_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, DENSE_128, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)


def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_512_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_512_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, DENSE_512, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)


def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, DENSE_1024, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)


def QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_64_BN_Dense_43(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_64_BN_Dense_43_ep_' + \
        str(EPOCHS)

    start = datetime.now()
    model = build_model(FILTER_32, KERNEL_SIZE_5, FILTER_64, KERNEL_SIZE_5,
                        FILTER_64, KERNEL_SIZE_3, DENSE_64, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    end = datetime.now()
    get_n_save_execution_time(start, end, TEST_NAME)




if __name__ == "__main__":

    data, labels = get_training_dataset()
    X_train, X_validation, y_train, y_validation = split_in_training_n_test(
        data, labels)

    X_test, y_test = get_testing_dataset()

    # QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_43(
    #     X_train, X_validation, y_train, y_validation, X_test, y_test)

    QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_64_BN_Dense_43(
        X_train, X_validation, y_train, y_validation, X_test, y_test)

    QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_128_BN_Dense_43(
        X_train, X_validation, y_train, y_validation, X_test, y_test)

    QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_256_BN_Dense_43(
        X_train, X_validation, y_train, y_validation, X_test, y_test)

    QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_512_BN_Dense_43(
        X_train, X_validation, y_train, y_validation, X_test, y_test)

    QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43(
        X_train, X_validation, y_train, y_validation, X_test, y_test)
