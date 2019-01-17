import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import chainer

def get_dataset():
    FILE_TRAIN_LABELS_PATH = "../Dataset/train-labels-idx1-ubyte.gz"
    FILE_TRAIN_IMAGES_PATH = "../Dataset/train-images-idx3-ubyte.gz"
    FILE_TEST_LABELS_PATH = "../Dataset/t10k-labels-idx1-ubyte.gz"
    FILE_TEST_IMAGES_PATH = "../Dataset/t10k-images-idx3-ubyte.gz"
    
    f = gzip.open(FILE_TRAIN_IMAGES_PATH,'r')
    f.read(16)
    num_train_images = 60000
    count_train_image_rows = 28
    count_train_image_cols = 28
    buf = f.read(count_train_image_rows * count_train_image_cols * num_train_images)
    train_data = np.frombuffer(buf, dtype=np.uint8)
    train_data = train_data.reshape(num_train_images, count_train_image_rows * count_train_image_cols) / 255

    train_labels = []

    f = gzip.open(FILE_TRAIN_LABELS_PATH,'r')
    f.read(8)
    num_train_labels = 60000
    for i in range(0, num_train_labels):
        buf = f.read(1)
        train_labels.append((np.frombuffer(buf, dtype=np.uint8).astype(np.int64)))

    train_labels = np.array(train_labels)

    onehot_encoder = OneHotEncoder(sparse=False)
    train_labels = train_labels.reshape(len(train_labels), 1)
    train_labels = onehot_encoder.fit_transform(train_labels)

    f = gzip.open(FILE_TEST_IMAGES_PATH,'r')
    f.read(16)
    num_test_images = 10000
    count_test_image_rows = 28
    count_test_image_cols = 28
    buf = f.read(count_test_image_rows * count_test_image_rows * num_test_images)
    test_data = np.frombuffer(buf, dtype=np.uint8)
    test_data = test_data.reshape(num_test_images, count_test_image_rows * count_test_image_rows) / 255

    test_labels = []

    f = gzip.open(FILE_TEST_LABELS_PATH,'r')
    f.read(8)
    num_test_labels = 10000
    for i in range(0, num_test_labels):
        buf = f.read(1)
        test_labels.append((np.frombuffer(buf, dtype=np.uint8).astype(np.int64)))

    test_labels = np.array(test_labels)

    onehot_encoder = OneHotEncoder(sparse=False)
    test_labels = test_labels.reshape(len(test_labels), 1)
    test_labels = onehot_encoder.fit_transform(test_labels)

    return train_data, train_labels, test_data, test_labels
