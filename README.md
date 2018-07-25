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
