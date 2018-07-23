#!/bin/bash

for f in emnist*.yml; do
    echo "Filename:" $f >> output.txt
    cat $f | grep "batch_size" >> output.txt
    cat $f | grep "number_of_workers" >> output.txt
    echo "" >> output.txt

    for i in {1..5}
    do
        echo "Round number" $i >> output.txt
        # Build and deploy the function to launch new pods
        faas-cli build -f $f --parallel=2 && faas-cli push -f $f --parallel=2 && faas-cli deploy -f $f

        # Let the old pods terminate and the new one launch
        sleep 200

        # Invoke the function
        echo | faas-cli invoke emnist-setup

        # Append the table of pods to the output file to have information about where each pod is executing
        kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train" >> output.txt

        # If all the pods terminated the execution, there are number of processes lines in the logs command with the word "Time" in it
        # Timeout after 2 hours, in case something wrong happens
        START=`date +%s`
        while [ $(( $(date +%s) - 7200 )) -lt $START ]; do
            CMD1=$(for server in $(kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train" | cut -d' ' -f1); do kubectl logs --namespace=openfaas-fn $server emnist-train; done | grep -c "Time")
            CMD2=$(for server in $(kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train" | cut -d' ' -f1); do kubectl logs --namespace=openfaas-fn $server emnist-train; done | grep -c "Worker number")
            if [ "$CMD1" = "$CMD2" ] && [ "$CMD1" > 0 ] ; then
                for server in $(kubectl get pods --namespace=openfaas-fn -o wide | grep "emnist-train" | cut -d' ' -f1); do kubectl logs --namespace=openfaas-fn $server emnist-train; done | grep "Time" >> output.txt
                break
            fi
            sleep 60
        done
        echo "" >> output.txt
    done
    echo "" >> output.txt
done
