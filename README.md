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

The full stack of tests is available in the `tests` folder and can be run from with the `test.sh` script. To include a test on the current execution of `test.sh` the corresponding file from `tests` should be copied into the home file. Diferent files have different values of `batch_size` and `number_of_pods`. After the required tests are in the project folder, they can be run using:
```bash
$ nohup test.sh &
```
The outputs are saved in an `output.txt` file.

### DownpoutSGD_global_client_nFetch10, emnist321.yml

| Round | Time (s)  | Iterations | 42 | 41 | 40 |
| ----- | --------- | ---------- | -- | -- | -- |
| 1     | 1415      | 5          |    |    | X  |
| 2     | 209       | 5          |  X |    |    |
| 3     | 770       | 5          |    | X  |    |
| 4     | 768       | 5          |    | X  |    |
| 5     | 210       | 5          |  X |    |    |
|       |           |            |    |    |    |
| Avg   | 674       | 5          |    |    |    |
| Stdev | 499       | 0          |    |    |    |

### DownpoutSGD_global_client_nFetch10, emnist323.yml

| Round | Time (s)  | Iterations  | 42 | 41 | 40 |
| ----- | --------- | ----------- | -- | -- | -- |
| 1     | 393       | 3           |    | X  |    |
|       | 394       | 3           |    | X  |    |
|       | 352       | 16          | X  |    |    |
| 2     | 620       | 11          |    |    | 1  |
|       | 595       | 10          | X  |    |    |
|       | 597       | 9           | X  |    |    |
| 3     | 1914      | 11          |    | X  |    |
|       | 1963      | 11          |    | X  |    |
|       | 1964      | 13          |    | Y  |    |
| 4     | 415       | 3           |    | X  |    |
|       | 415       | 3           |    | X  |    |
|       | 363       | 16          | X  |    |    |
| 5     | 355       | 17          | X  |    |    |
|       | 473       | 4           |    | X  |    |
|       | 474       | 4           |    | X  |    |
|       |           |             |    |    |    |
| Avg   | 752       | 9           |    |    |    |
| Stdev | 624       | 5           |    |    |    |

The full set of results is available in the `test` folder.

## Branches

The branch naming follows the convention name `algotithmName_databaseType_args_type`, where `algorithm` might be DownpourSDG, `databaseType` global or local (with synchronization in the background) and `args` can be the nFetch or nPull parameters. The types are the following:

1. _Single function training_ means that a single function is responsible for loading the data, initialising and training the network.

The remaining need 2 different functions deployed on OpenFaaS. A _setup_ function that initialises the network and the state, and then calls a _train_ function, that trains the network with the data.

2. _One function per iteration_ means that each iteration of SGD is run in a separate function. If the training takes 50 iterations, at the end of each iteration the _train_ function requests the gateway to schedule the next invocation of the function. If the data is divided into M parts, each of the parts executes one function per iteration.

3. _Setup + loop in one function_ means that the entire training (all iterations) are run in a single function. The _setup_ function initialises the network and then calls M _train_ functions depending on the number of machines in the cluster.

Type 3 has the added benefits of natural scalability and fault tolerance.

## Tasks

- [x] Write a proper test suite to benchmark the system
- [x] Find a bigger dataset, in the order of 1GB, and adjust it to this particular problem
- [x] Explore data parallelism in the larger dataset, using the function parallelism native in OpenFaaS to parallelise separate batches of data (Average parameters)
- [x] Adding a Redis / Memcached pod on the Kubernetes cluster. The data will start being stored and retrieved from this datastore rather than passing via HTTP requests
- [x] Building a Python decorator that makes the database calls without the programmer having to hard code them
- [x] Add affinity between the OpenFaaS and the Redis / Memcached pods to have the data used by an OpenFaaS worker stored in that same worker
- [x] Explore data parallelism in the larger dataset, using the function parallelism native in OpenFaaS to parallelise separate batches of data (Downpour SGD)
- [ ] Synchronize multiple memcached machines in the background
- [ ] Update only the changed parameters in each minibatch
- [ ] Add fault tolerance to the system
- [ ] Repeat tests with more machines (Kubernetes scheduling problems)
- [ ] Deal with specific bottlenecks of the system
- [ ] Show some aditional use cases
