import os
import requests
import sys
from urllib.parse import parse_qs
import numpy as np
from scipy import io as spio
import keras
import time

import bmemcached


def tanh(x):
    """Compute hyperbolic tangent element-wise.

    Args:
        x (array_like): Input array.

    Returns:
        ndarray: The corresponding hyperbolic tangent values.
    """
    return np.tanh(x)


def tanh2deriv(x):
    """Compute hyperbolic tangent derivative element-wise.

    Args:
        x (array_like): Input array.

    Returns:
        ndarray: The corresponding hyperbolic tangent derivative values.
    """
    return 1 - (x**2)


def softmax(x):
    """Computes softmax activations.

    Args:
        x (array_like): Input array.

    Returns:
        ndarray: The corresponding softmax values.
    """
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def stateful(*decorator_args):
    """Decorator to load and persist values of lambda function.
    Args:
        decorator_args (list(str)): Name of the values to R/W.
    Returns:
        wrap(f): A function used to wrap the function to be decorated.
    """

    def wrap(f):
        # Everything before decoration happens here
        client = bmemcached.Client((os.getenv("MEMCACHED_SERVICE_HOST"),
                                    int(os.getenv("MEMCACHED_SERVICE_PORT"))))

        def wrapped_f(*args):
            # After decoration
            # Before function
            state = ()

            for arg in decorator_args:
                state += (client.get(arg), )

            # The last value returned by f should be the decorator_args
            return_vals = f(*args, state)

            # After function
            state = return_vals[-1]
            i = 0
            for arg in decorator_args:
                client.set(arg, state[i].tobytes())
                i += 1

            return return_vals

        return wrapped_f

    return wrap


def load_dataset(worker_id, number_of_workers):
    """Load emnist-bymerge dataset from disk and parse it.

    Args:
        worker_id (int): ID of the current worker loading the dataset.
        number_of_workers (int):
            Total number of workers working with the dataset.

    Returns:
        images (nparray): The array of train images in the dataset
        labels (nparray): The corresponding train labels
        test_images (nparray): The array of test images in the dataset
        test_labels (nparray): The corresponding test labels
    """
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


@stateful('weights_0_1', 'weights_1_2')
def train_minibatch(i, pixels_per_image, hidden_size, num_labels, batch_size,
                    images, dropout_percent, correct_cnt, labels, alpha,
                    state):
    """Train a minibatch of data with SGD.

    Args:
        i (int): number of the current minibatch
        pixels_per_image (int): number of pixels in each image
        hidden_size (int): size of the hidden layer of the network
        num_labels (int):
            total number of possible classifications of the dataset
        batch_size (int): size of each minibatch (number of images)
        images (nparray): array of train images
        dropout_percent (float):
            percentage of parameters ignored at the current stage
        correct_cnt (int):
            number of correct classifications so far in the training
        labels (nparray): array of train labels
        alpha (int): alpha value for the SGD algorithm
        state: Variable length argument list,
            one for each variable in the decorator.

    Returns:
        correct_cnt, layer_0, layer_1, weights_0_1, weights_1_2
        correct_cnt (int):
            number of correct classifications so far in the training
        layer_0 (nparray): zero-th layer of the current network
        layer_1 (nparray): first layer of the current network
        weights_0_1 (nparray): value of the parameters between layer 0 and 1
        weights_1_2 (nparray): value of the parameters between layer 1 and 2
    """
    # *args will be weights_0_1 and weights_1_2 from the decorator
    weights_0_1 = np.frombuffer(state[0]).reshape(pixels_per_image,
                                                  hidden_size)
    weights_1_2 = np.frombuffer(state[1]).reshape(hidden_size, num_labels)

    weights_0_1.flags.writeable = True
    weights_1_2.flags.writeable = True

    batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
    # 2. PREDICT & COMPARE: Make a Prediction,
    #    Calculate Output Error and Delta
    layer_0 = images[batch_start:batch_end]
    layer_1 = tanh(np.dot(layer_0, weights_0_1))
    dropout_mask = np.random.randint(2, size=layer_1.shape)

    layer_1 *= np.random.binomial([np.ones(
        (len(layer_1), hidden_size))], 1 - dropout_percent)[0] * (
            1.0 / (1 - dropout_percent))
    layer_2 = softmax(np.dot(layer_1, weights_1_2))

    for k in range(batch_size):
        correct_cnt += int(
            np.argmax(layer_2[k:k + 1]) == np.argmax(labels[
                batch_start + k:batch_start + k + 1]))

    layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (
        batch_size * layer_2.shape[0])
    # 3. LEARN: Backpropagate From layer_2 to layer_1
    layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
    layer_1_delta *= dropout_mask

    # 4. LEARN: Generate Weight Deltas and Update Weights FIXME
    weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
    weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    return correct_cnt, layer_0, layer_1, (weights_0_1, weights_1_2)


def handle(req):
    """handle a request to the function

    Args:
        req (str): request body
    """
    # Print next line for debug only
    # sys.stderr.write(str(os.environ))
    client = bmemcached.Client((os.getenv("MEMCACHED_SERVICE_HOST"),
                                int(os.getenv("MEMCACHED_SERVICE_PORT"))))

    # get the query string from the http request
    qs = parse_qs(os.getenv("Http_Query"))

    # receives the parameters needed for the training
    alpha = float(client.get('alpha'))
    hidden_size = int(client.get('hidden_size'))
    pixels_per_image = int(client.get('pixels_per_image'))
    num_labels = int(client.get('num_labels'))
    batch_size = int(client.get('batch_size'))
    dropout_percent = float(client.get('dropout_percent'))
    number_of_workers = int(client.get('number_of_workers'))
    worker_id = int(qs["worker_id"][0])

    accuracy = float(client.get('accuracy%d'.format(worker_id)))
    iteration = int(client.get('iteration%d'.format(worker_id)))

    # set the random seed
    np.random.seed(1)

    # load the dataset
    images, labels, test_images, test_labels = load_dataset(
        worker_id, number_of_workers)

    if iteration == 1:
        sys.stderr.write("Worker number " + str(worker_id))

    if accuracy < 0.75:

        correct_cnt = 0

        for i in range(int(len(images) / batch_size)):
            correct_cnt, layer_0, layer_1, (weights_0_1,
                                            weights_1_2) = train_minibatch(
                                                i, pixels_per_image,
                                                hidden_size, num_labels,
                                                batch_size, images,
                                                dropout_percent, correct_cnt,
                                                labels, alpha)

        test_correct_cnt = 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i + 1]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = softmax(np.dot(layer_1, weights_1_2))
            test_correct_cnt += int(
                np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

        accuracy = test_correct_cnt / float(len(test_images))
        iteration += 1

        client.set('accuracy%d'.format(worker_id), accuracy)
        client.set('iteration%d'.format(worker_id), iteration)

        # uses a default of "gateway" for when "gateway_hostname" is not set
        gateway_hostname = os.getenv("gateway_hostname", "gateway")
        # set the variables to send to the first iteration of the training loop
        payload = {'worker_id': worker_id}
        # make the request to call the emnist-train function HACKY timeout
        try:
            requests.get(
                "http://" + gateway_hostname + ":31112/function/emnist-train",
                params=payload,
                timeout=0.1)
        except requests.exceptions.ReadTimeout:
            pass

    else:
        start = float(client.get('start'))
        end = time.time()
        sys.stderr.write("\n" + "Worker ID: " + str(worker_id) +
                         " Iterations: " + str(iteration) + " Time: " +
                         str(end - start) + " Name: " + os.getenv("HOSTNAME"))
