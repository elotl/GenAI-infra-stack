# Setting up LLM observability for your Question-Answering in a box deployment


## Setup Phoenix on your K8s cluster

```shell
cd GenAI-infra-stack/dockers/llm.phoenix.service
```

```shell
kubectl apply -f  phoenix.yaml
```


```shell
kubectl port-forward service/phoenix 6006:443
```
