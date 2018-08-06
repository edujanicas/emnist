import numpy as np
import os
import requests
import sys
import time

import pylibmc


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    sys.stderr.write(str(os.environ))

    # random seed set to 1 for debug purposes
    np.random.seed(1)

    # get a connection to the memcached database
    client = pylibmc.Client(
        ['146.179.131.181:11211'],
        binary=True,
        behaviors={
            "tcp_nodelay": True,
            "ketama": True
        })

    # 1. Initialize the Network's Weights and Data
    # TODO: Switch values to become env vars
    start = time.time()

    alpha = float(os.getenv("alpha"))
    hidden_size = int(os.getenv("hidden_size"))
    pixels_per_image = int(os.getenv("pixels_per_image"))
    num_labels = int(os.getenv("num_labels"))
    batch_size = int(os.getenv("batch_size"))
    dropout_percent = float(os.getenv("dropout_percent"))
    number_of_workers = int(os.getenv("number_of_workers"))

    weights_0_1 = 0.02 * np.random.random(
        (pixels_per_image, hidden_size)) - 0.01
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    client.set('weights_0_1', weights_0_1.tobytes())
    client.set('weights_1_2', weights_1_2.tobytes())
    client.set('updated', 0)
    client.set('alpha', alpha)
    client.set('hidden_size', hidden_size)
    client.set('pixels_per_image', pixels_per_image)
    client.set('num_labels', num_labels)
    client.set('batch_size', batch_size)
    client.set('dropout_percent', dropout_percent)
    client.set('number_of_workers', number_of_workers)
    client.set('start', start)

    # uses a default of "gateway" for when "gateway_hostname" is not set
    gateway_hostname = os.getenv("gateway_hostname", "gateway")

    for worker_id in range(number_of_workers):
        # set the variables to send to the first iteration of the training loop
        payload = {'worker_id': worker_id}
        client.set('accuracy' + str(worker_id), 0)
        client.set('iteration' + str(worker_id), 1)
        # make the request to call the emnist-train function HACKY timeout
        try:
            requests.get(
                "http://" + gateway_hostname + ":31112/function/emnist-train",
                params=payload,
                timeout=0.1)
        except requests.exceptions.ReadTimeout:
            pass
