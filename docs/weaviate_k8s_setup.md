# Install Weaviate

```bash
helm repo add weaviate https://weaviate.github.io/weaviate-helm
helm repo update
```

## Create the weaviate namespace

```bash
kubectl create namespace weaviate
```

## Install via the weaviate helm chart

```bash
cd GenAI-infra-stack/demo/weaviate
```


```bash
helm upgrade --install \
  "weaviate" \
  weaviate/weaviate \
  --namespace "weaviate" \
  --values ./values.yaml
```

## Check successful install

```bash
% kubectl get pods -n weaviate      
NAME         READY   STATUS    RESTARTS   AGE
weaviate-0   1/1     Running   0          25s
```

```bash
% kubectl get statefulset -n weaviate
NAME       READY   AGE
weaviate   1/1     31s
```

## Check connectivity to weaviate using a Python client


```bash
kubectl port-forward svc/weaviate-grpc 50051:50051 -n weaviate
kubectl port-forward svc/weaviate 8080:80 -n weaviate
```


```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run test_weaviate.py      
Connection to weaviate successful: True
```

