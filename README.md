# EMNIST Neural network

This repo contains the functions to train a neural network with the emnist dataset on top of OpenFaaS.

## Getting started

### Prerequisites

These functions take advantage of the persistent state of lambdas when OpenFaaS is deployed with [faas-state](https://github.com/edujanicas/faas-netes "faas-state"). Check on the repo for instructions on how to install.

[faas-cli](https://github.com/openfaas/faas-cli "faas-cli") must be installed in the developer machine to build, push and deploy the function. The `OPENFAAS_URL` environment variable must be set to the respective OpenFAAS cluster. Docker must be running on the developer machine.

Access to the OpenFAAS cluster must also be in place in order to read the results.

### Installing

On the project home folder, run the following command:
```bash
$ faas-cli build -f emnist.yml --parallel=2 && faas-cli push -f emnist.yml --parallel=2 && faas-cli deploy -f emnist.yml
```

After this step you should be able to invoke it by executing:
```bash
$ echo | faas-cli invoke emnist-setup
```

## Running the tests

The following results are obtained in a 3 machine Kubernetes cluster with Intel(R) Xeon(R) CPU E3-1220 v3 @ 3.10GHz and 16GB of RAM per server. The following commands are executed on the k8s master.

To scale the deployment to the number of desired machines, execute:
```bash
$ kubectl scale deployment emnist-train --replicas=3 --namespace=openfaas-fn
```

To see the names of the pods executing the function, run:
```bash
$ kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train"
```

To see the results of the training in the logs, run:
```bash
$ for server in $(kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train" | cut -d' ' -f1); do kubectl logs --namespace=openfaas-fn $server emnist-train; done
```

### Setup + loop in one function (32 batch size, update before each mini-batch)

|    N    |    Target-accuracy    |    Iterations    |    Time    |  
| ----- | -------------------- | ------------- | -------------- |
| 1        | 0.75                             | 5                    | 298 - 1567       |
| 3        | 0.75                            | 3 - 15             | 541         |
| 6        | 0.75                            | 9 - 23             | 1288      |

With N = 1, the time varies from 298s if the memcached pod is in the same machine that executes the function, or 1567s on a different machine.

### Setup + loop in one function (100 batch size)
|    N    |    Target-accuracy    |    Iterations    |    Time    |  
| ----- | -------------------- | ------------- | --------- |
| 1        | 0.75                           | 44                  | 1430          |
| 3        | 0.75                          | 41                   | 2496        |
| 6        | 0.75                          |                       |                 |