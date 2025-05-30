# Using this Text Summarizer UI


## Deploy the sample Ray Service


Text Summarizer Ray Service: https://docs.ray.io/en/latest/cluster/kubernetes/examples/text-summarizer-rayservice.html


Steps to install the Ray Service:


## Install Ray Operator

Install helm chart

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0
```

Check that the kuberay operator was deployed:

```bash
% helm ls
NAME            	NAMESPACE	REVISION	UPDATED                            	STATUS  	CHART                 	APP VERSION
kuberay-operator	default  	1       	2025-05-30 10:34:42.20226 -0700 PDT	deployed	kuberay-operator-1.3.0	           
```

```bash
% kubectl get pods
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-66d848f5cd-6tc9z   1/1     Running   0          42s
```


## Install text summarizer Ray Service

```bash
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/samples/ray-service.text-summarizer.yaml
```


Check that the summarizer service and pods are created:

```bash
% kubectl get svc
NAME                                        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                                         AGE
kuberay-operator                            ClusterIP   10.100.10.124   <none>        8080/TCP                                        2m32s
kubernetes                                  ClusterIP   10.100.0.1      <none>        443/TCP                                         50d
text-summarizer-raycluster-8xmdp-head-svc   ClusterIP   None            <none>        10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP   74s
selvik@Selvis-MacBook-Pro llm.summarizerui.service % kubectl get pods
NAME                                                      READY   STATUS              RESTARTS   AGE
kuberay-operator-66d848f5cd-6tc9z                         1/1     Running             0          2m44s
text-summarizer-raycluster-8xmdp-gpu-group-worker-qmf99   0/1     Pending             0          86s
text-summarizer-raycluster-8xmdp-head-t2vfd               0/1     ContainerCreating   0          86s
```



