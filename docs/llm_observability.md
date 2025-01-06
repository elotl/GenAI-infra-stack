# Setting up LLM observability for your Question-Answering in a box deployment


## Setup Phoenix on your K8s cluster

```shell
cd GenAI-infra-stack/dockers/llm.phoenix.service
```

```shell
kubectl apply -f  phoenix.yaml
```

Check that all the related resources have been started up:

```shell
% kubectl get all -n phoenix
```

```shell
NAME            READY   STATUS              RESTARTS   AGE
pod/phoenix-0   0/1     ContainerCreating   0          8s

NAME              TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
service/phoenix   ClusterIP   10.100.4.179   <none>        6006/TCP   8s

NAME                       READY   AGE
statefulset.apps/phoenix   0/1     8s
```


In order to access the Phoenix UI, port-forward in a separate terminal:
```shell
kubectl port-forward -n phoenix service/phoenix 6006:6006
```
Don't miss the namespace `-n phoenix` since the service is in the phoenix namespace and not the default namespace

Open this URL in a local browser:
http://localhost:6006/projects

Add a screenshot on what the Phoenix UI will look like.


