import numpy as np
import os
import requests

from pymemcache.client import base

def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    np.random.seed(1)
    client = base.Client((os.getenv("MEMCACHED_SERVICE_HOST"), int(os.getenv("MEMCACHED_SERVICE_PORT"))))

    ### 1. Initialize the Network's Weights and Data

    alpha = 0.1
    iterations = 55
    hidden_size = 100
    pixels_per_image = 784
    num_labels = 47
    batch_size = 100
    dropout_percent = 0.2

    weights_0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    # set the variables to send to the first iteration of the training loop
    payload = {'alpha': alpha,
               'iterations': iterations,
               'hidden_size': hidden_size,
               'pixels_per_image': pixels_per_image,
               'num_labels': num_labels,
               'batch_size': batch_size,
               'dropout_percent':dropout_percent
               }

    # uses a default of "gateway" for when "gateway_hostname" is not set
    gateway_hostname = os.getenv("gateway_hostname", "gateway")

    client.set('weights_0_1', weights_0_1.tobytes())
    client.set('weights_1_2', weights_1_2.tobytes())

    # make the request to call the emnist-train function HACKY timeout
    try:
        requests.get(
            "http://" + gateway_hostname + ":31112/function/emnist-train",
            params=payload,
            timeout=0.1
        )
    except requests.exceptions.ReadTimeout:
        pass