# EMNIST Neural network

This repo contains the functions to train a neural network with the emnist dataset on top of OpenFaaS deployed with [faas-state](https://github.com/edujanicas/faas-netes "faas-state").

## Build
`$ faas-cli build -f emnist.yml --parallel=2 && faas-cli push -f emnist.yml --parallel=2 && faas-cli deploy -f emnist.yml`

## Run
`$ echo | curl http://146.179.131.181:31112/function/emnist-setup`

## Results
### Setup + loop in one function (32 batch size, update before each mini-batch)

|    N    |    Target-accuracy    |    Iterations    |    Time    |  
| ----- | -------------------- | ------------- | --------- |
| 1        | 0.75                             | 5                    | 1567       |
| 3        | 0.75                            | 3 - 15             | 541         |
| 6        | 0.75                            | 9 - 23             | 1288      |

### Setup + loop in one function (100 batch size)
|    N    |    Target-accuracy    |    Iterations    |    Time    |  
| ----- | -------------------- | ------------- | --------- |
| 1        | 0.7                             | 20                   | 439        |
| 3        | 0.7                             | 60                   | 708        |
| 6        | 0.7                             | 120                 |  1159      |