import os
import sys
from urllib.parse import parse_qs
import numpy as np
from scipy import io as spio
import keras

from pymemcache.client import base


def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output**2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def load_dataset(worker_id, number_of_workers):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'data', 'emnist-bymerge.mat')
    emnist = spio.loadmat(path)

    # load training dataset
    images = emnist["dataset"][0][0][0][0][0][0]
    images = images.astype(np.float32)

    # load training labels
    labels = emnist["dataset"][0][0][0][0][0][1]

    # load test dataset
    test_images = emnist["dataset"][0][0][1][0][0][0]
    test_images = test_images.astype(np.float32)

    # load test labels
    test_labels = emnist["dataset"][0][0][1][0][0][1]

    # normalize
    images /= 255
    test_images /= 255

    # labels should be onehot encoded
    labels = keras.utils.to_categorical(labels, 47)
    test_labels = keras.utils.to_categorical(test_labels, 47)

    # crop the input
    images = images[int((len(images) / number_of_workers) * worker_id):int(
        (len(images) / number_of_workers) * (worker_id + 1) - 1)]
    labels = labels[int((len(labels) / number_of_workers) * worker_id):int(
        (len(labels) / number_of_workers) * (worker_id + 1) - 1)]

    return images, labels, test_images, test_labels


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    # Print next line for debug only
    sys.stderr.write(str(os.environ))

    client = base.Client((os.getenv("MEMCACHED_SERVICE_HOST"),
                          int(os.getenv("MEMCACHED_SERVICE_PORT"))))

    # get the query string from the http request
    qs = parse_qs(os.getenv("Http_Query"))

    # receives the parameters needed for the training
    alpha = float(client.get('alpha'))
    iterations = int(client.get('iterations'))
    hidden_size = int(client.get('hidden_size'))
    pixels_per_image = int(client.get('pixels_per_image'))
    num_labels = int(client.get('num_labels'))
    batch_size = int(client.get('batch_size'))
    dropout_percent = float(client.get('dropout_percent'))
    number_of_workers = int(client.get('number_of_workers'))
    worker_id = int(qs["worker_id"][0])

    weights_0_1 = np.frombuffer(client.get('weights_0_1')).reshape(
        pixels_per_image, hidden_size)
    weights_1_2 = np.frombuffer(client.get('weights_1_2')).reshape(
        hidden_size, num_labels)
    weights_0_1.flags.writeable = True
    weights_1_2.flags.writeable = True

    # set the random seed
    np.random.seed(1)

    # load the dataset
    images, labels, test_images, test_labels = load_dataset(
        worker_id, number_of_workers)

    for iter in range(iterations):

        correct_cnt = 0

        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            # 2. PREDICT & COMPARE: Make a Prediction,
            #    Calculate Output Error and Delta
            layer_0 = images[batch_start:batch_end]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)

            layer_1 *= np.random.binomial(
                [np.ones((len(layer_1), hidden_size))],
                1 - dropout_percent)[0] * (1.0 / (1 - dropout_percent))
            layer_2 = softmax(np.dot(layer_1, weights_1_2))

            for k in range(batch_size):
                correct_cnt += int(
                    np.argmax(layer_2[k:k + 1]) == np.argmax(labels[
                        batch_start + k:batch_start + k + 1]))

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (
                batch_size * layer_2.shape[0])
            # 3. LEARN: Backpropagate From layer_2 to layer_1
            layer_1_delta = layer_2_delta.dot(
                weights_1_2.T) * tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask

            # 4. LEARN: Generate Weight Deltas and Update Weights
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        test_correct_cnt = 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i + 1]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = softmax(np.dot(layer_1, weights_1_2))
            test_correct_cnt += int(
                np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

        sys.stderr.write("\n" + "I:" + str(iter) + " Test-Acc:" +
                         str(test_correct_cnt / float(len(test_images))) +
                         " Train-Acc:" + str(correct_cnt / float(len(images))))

        if client.get('updated') == 0:
            client.set('weights_0_1', weights_0_1.tobytes())
            client.set('weights_1_2', weights_1_2.tobytes())
            client.set('updated', 1)
        else:
            weights_0_1_global = np.frombuffer(
                client.get('weights_0_1')).reshape(pixels_per_image,
                                                   hidden_size)
            weights_1_2_global = np.frombuffer(
                client.get('weights_1_2')).reshape(hidden_size, num_labels)

            a_0_1 = np.array([weights_0_1, weights_0_1_global])
            a_1_2 = np.array([weights_1_2, weights_1_2_global])
            averaged_0_1 = np.average(a_0_1, axis=0)
            averaged_1_2 = np.average(a_1_2, axis=0)

            client.set('weights_0_1', averaged_0_1.tobytes())
            client.set('weights_1_2', averaged_1_2.tobytes())