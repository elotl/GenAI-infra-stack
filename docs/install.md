# LLM Chat in a Box POC, v0.1.4, 10/22/24

[K8s](https://kubernetes.io/) + [Luna](https://docs.elotl.co/luna/intro/) + [KubeRay](https://docs.ray.io/en/master/cluster/kubernetes/getting-started.html) + [RayService](https://docs.ray.io/en/master/cluster/kubernetes/getting-started/rayservice-quick-start.html) + [vLLM](https://docs.vllm.ai/en/stable/) + Open Source LLM + [RayAutoscaler](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/configuring-autoscaling.html) + [Retrieval Augmented Generation using FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)  

## Overview
This POC is intended to allow you to easily deploy and use a working state-of-the-art chat serving platform for an open-source LLM model via automatically scaling your EKS, GKE, or AKS cloud Kubernetes cluster with economical compute instances.  And to easily tear down the deployed serving platform when desired.

## Cluster Setup Summary

Run w/Luna on K8s w/L4 (EKS,GKE) & A10 (AKS) w/GPU quota + specified Nvidia GPU drivers

* Luna-1.2.4, EKS, us-west-2, K8s v1.30.2, w/K8s Nvidia daemonset from
[https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml](https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
)
* Luna-1.2.4, GKE, us-central1, K8s v1.29.7, w/GCP Nvidia daemonset from
[https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml](https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml)
* Luna-1.2.5pre, AKS, east-us, K8s v1.30.3, w/K8s Nvidia gpu-operator from
```sh
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator
```

## Install Infrastructure Tools

### Install Luna Cluster Autoscaler on Cloud K8s Cluster

On existing cloud K8s cluster, install Luna as per cloud K8s in the [Luna docs](https://docs.elotl.co/luna/intro/).
[Download Free trial here](https://www.elotl.co/luna-free-trial.html).
For EKS: need to specify larger EBS size w/Luna aws.blockDeviceMapping option
Download block_device_mapping.json and when deploying Luna, include --additional-helm-values “--set-file aws.blockDeviceMappings=<path>/block_device_mapping.json”

### Install KubeRay Operator to manage Ray on Cloud K8s Cluster
```sh
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0-rc.0
```
```sh
# Small resource footprint; Installed on static, i.e., non-Luna-allocated resources
```

## Install Model Serve Stack
You can choose to install the RayService w/vLLM + Open Source Model Serve Stack either without or with the Ray Autoscaler, as described in the 2 subsections for each of the two models below.  If you install it w/o the Ray Autoscaler, the model serve stack will come up more quickly, but will have a fixed number of workers, configured as 1.  If you install it with the Ray Autoscaler, the model serve stack will start with 0 workers, will scale to 1 worker as the RayService is activated, and will scale to more workers as needed to handle the query load, configured w/a max of 4.

### [MosaicML Open Source Model](https://huggingface.co/mosaicml/mpt-7b-chat)
Install RayService w/vLLM + MosaicML OS Model w/o Ray Autoscaler

```sh
kubectl apply -f ray-service.llm.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.yaml
# Large resource footprint; Installed on Luna-allocated resources
# Takes 10-15m: add nodes + large image + vLLM update + Ray setup + model download
# Wait for svc/llm-model-serve-serve-svc to be available [This is the last of the 3 services started]
```

Install RayService w/vLLM + MosaicML OS Model w/ Ray Autoscaler

```sh
kubectl apply -f ray-service.llm.autoscale.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.autoscale.yaml
# Large resource footprint; Installed on Luna-allocated resources
# Takes 10-15m: add nodes + large image + vLLM update + Ray setup + model download
# Wait for svc/llm-model-serve-serve-svc to be available [This is the last of the 3 services started]
```

### [Microsoft Open Source Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
Install RayService w/vLLM + Microsoft OS Model w/o Ray Autoscaler

```sh
kubectl apply -f ray-service.llm.Phi-3-mini-4k-instruct.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.Phi-3-mini-4k-instruct.yaml
# Large resource footprint; Installed on Luna-allocated resources
# Takes 10-15m: add nodes + large image + vLLM update + Ray setup + model download
# Wait for svc/llm-model-serve-serve-svc to be available [This is the last of the 3 services started]
```

Install RayService w/vLLM + Microsoft OS Model w/ Ray Autoscaler

```sh
kubectl apply -f ray-service.llm.Phi-3-mini-4k-instruct.autoscale.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.Phi-3-mini-4k-instruct.autoscale.yaml
# Large resource footprint; Installed on Luna-allocated resources
# Takes 10-15m: add nodes + large image + vLLM update + Ray setup + model download
# Wait for svc/llm-model-serve-serve-svc to be available [This is the last of the 3 services started]
```

## Model Serve

### Run Port-forward for Model Endpoint
```sh
kubectl port-forward svc/llm-model-serve-serve-svc 8000:8000
```

### Query Model Endpoint

MosaicML Open Source Model

```sh
python query.py
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/query.py
# Run in separate terminal window from port-forward command
# Requires “pip install openai”
# Prompts user for query
```

Microsoft Open Source Model

```sh
python query.py
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/query.py
# Run in separate terminal window from port-forward command
# Requires “pip install openai”
# Requires “export MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
# Prompts user for query
```

### Experiment with Scaling

If you installed the model serve stack with the Ray Autoscaler, you can experiment with scaling by presenting sufficient query load.  Running the following shell script presented enough load for the Ray Autoscaler to increase the number of workers from 1 to 2, for which the Luna Autoscaler added an additional GPU node to the cluster.

```sh
./loadtest.sh
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/loadtest.sh
# Run in clone of repo with question.txt available in the same directory
# Use control-C to stop the script
```

After the load script was stopped and a period of time elapsed, the Ray Autoscaler reduced the worker count back to 1, and the Luna Autoscaler removed the additional node from the cluster.

## Retrieval Augmented Generation (RAG) using FAISS

In this section, we illustrate how LLM chat can be extended to work with custom datasets using the technique of Retrieval Augmented Generation. If you do not want to incorporate your custom datasets during the LLM chat you can skip this section. 

In order to use RAG, please follow the instructions in all of these prior sections before you follow the instructions in this section:
* Cluster Setup Summary
* Install Infrastructure Tools
* Install Luna Cluster Autoscaler on Cloud K8s Cluster
* Install KubeRay Operator to manage Ray on Cloud K8s Cluster
* Install Model Serve Stack
* Model Serve

In this section, we provide an example of storing your RAG dataset and the resulting Vector Store on AWS-specific S3 storage.  

### Setup RAG input dataset

In order to create the RAG dataset, we will run a Kubernetes job that will retrieve text documents from an S3 bucket, convert each text file into a vector embedding and save these embeddings in a Vector store. For the purpose of this POC, we use FAISS (Facebook’s Similarity Search) library to create both the embeddings and the VectorStore file. Please follow the instructions below setup the RAG dataset as well as the configure the parameters needed to run the vector Store creation Kubernetes job.


1. Create an S3 bucket and a folder (prefix) within it. Upload all the text documents that you would like to use as your RAG dataset into this folder.
Use can use the instructions here to create an S3 bucket: [Creating a S3 bucket in AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html#creating-bucket) and the instructions here to create a folder within this bucket:
[Folder creation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html#create-folder)

1. Create a local file with these environment variables exported with suitable values:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY: These AWS access credentials should have permissions to read from and write to the S3 bucket created in the previous step. 
If you would like to use your AWS CLI setup locally, you can use these commands to create these environment variables. Please note that these access credentials will not be limited to the minimal S3 bucket read and write permissions that are needed for setting up RAG. It is only provided here for ease of use (and is not meant for a production use).
```sh
export AWS_ACCESS_KEY_ID=$(grep aws_access_key_id ~/.aws/credentials | awk '{print $3}')
export AWS_SECRET_ACCESS_KEY=$(grep aws_secret_access_key ~/.aws/credentials | awk '{print $3}')
```
		
1. VECTOR_DB_INPUT_TYPE: Set this to a value of “text-docs" if the RAG dataset will be a set of documents in text format. Alternatively, you can set it to a value of “sitemap”, if the RAG dataset will need to be built from documents hosted on a website. For e.g: 
```sh
export VECTOR_DB_INPUT_TYPE=text-docs
```

1. VECTOR_DB_INPUT_ARG: 
If the VECTOR_DB_INPUT_TYPE value is “text-docs”, then this env variable will be set to the value of the folder or prefix name within an S3 bucket where the text documents will be uploaded by the customer. 
If the VECTOR_DB_INPUT_TYPE value is “sitemap”, then this env variable will be set to the URL value of sitemap of a website whose pages will be used as the RAG dataset.

1. VECTOR_DB_S3_BUCKET: Name of the S3 bucket that will contain the input dataset to be used for the RAG as well as RAG vector datastore. Please note that when VECTOR_DB_INPUT_TYPE value is “sitemap”, there is no input dataset that is needed to be uploaded to the S3 bucket. This is because the sitemap URL will be parsed by the Kto retrieve the dataset.

1. VECTOR_DB_S3_FILE: Name of the vector DB file that will be created by Elotl and saved in the provided S3 bucket.

1. MODEL_ID = [ microsoft/Phi-3-mini-4k-instruct | mosaicml/mpt-7b-chat ] Select the LLM model that is to be used. You can read about these two models here:
[https://huggingface.co/microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[https://huggingface.co/mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)

1. MODEL_LLM_SERVER_URL: Set this env var to the value http://llm-model-serve-serve-svc.<namespace-of-serveragllm>.svc.cluster.local:8000	
Please replace <namespace-of-serveragllm> with “default” if you plan to follow the remaining instructions in this RAG section without any changes.  In case you will be deploying the RAG LLM service in a custom namespace, then please replace <namespace-of-serveragllm> with the name of the custom namespace.
 
Source all these environment variables to your local shell. 

Note: If you chose to work with “text documents” for your RAG dataset, your documents would be made available at this S3 URI:
s3://<VECTOR_DB_S3_BUCKET>/<VECTOR_DB_INPUT_ARG>/ 

We will now setup some environment variables that are needed to enable us to customize how the Vector Store creation and RAG querying is performed.

```sh
# LLM querying configurable parameters:
    MODEL_ID (DEFAULT=mosaicml/mpt-7b-chat)
    RELEVANT_DOCS (DEFAULT = 2)
    MAX_TOKENS (DEFAULT=128)
    MODEL_TEMPERATURE (DEFAULT=0.01)
```

```sh
# Vector Store creation' configurable parameters:
    EMBEDDING_CHUNK_SIZE (DEFAULT=1000)
    EMBEDDING_CHUNK_OVERLAP (DEFAULT=100)
    EMBEDDING_MODEL_NAME (DEFAULT=sentence-transformers/all-MiniLM-L6-v2)
```

If You decide to pass to vector db creation a file created using our process_jira_tickets.py script set the following to "True".
```sh
    IS_JIRA_MODE (DEFAULT="False")
```

### Sample RAG Dataset
As an example of a RAG dataset, you could use this subset of Wikipedia docs: [https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
This dataset is accompanied with a number of Questions and Answers that can be used to validate RAG functionality.

You can use this command to download this dataset:
```sh
git clone https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia
```


## Creation of the Vector store
The vector Store can be created in your S3 bucket by running this Kubernetes job on your cluster. For the purpose of this POC, the default namespace is being used. Alternate namespaces can be used if desired. You can download createvdb.yaml from here: 
[https://github.com/elotl/GenAI-infra-stack/blob/main/demo/llm.vdb.service/createvdb.yaml](https://github.com/elotl/GenAI-infra-stack/blob/main/demo/llm.vdb.service/createvdb.yaml)

```sh
envsubst < createvdb.yaml | kubectl apply -f -
```

Ensure that the k8s job and corresponding pod are running as expected:

```sh
% kubectl get jobs
NAME         	STATUS	COMPLETIONS   DURATION   AGE
createvectordb   Running   0/1       	5s     	5s
```

```sh
% kubectl get pods
NAME                                                  	READY   STATUS	RESTARTS   AGE
createvectordb-kzrw6                                  	1/1 	Running   0      	118s
```

This will take a few minutes to complete. The logs in the above pod will end with these messages.

```sh
...SNIP...
Downloaded file, mini-rag-wikipedia-input/S10_set6_a9.txt.clean successfully to directory, /tmp/selvi-s3-rag-wikipedia
Downloaded file, mini-rag-wikipedia-input/S10_set6_topics.txt successfully to directory, /tmp/selvi-s3-rag-wikipedia
Number of files downloaded is 165, local tmp dir is /tmp/selvi-s3-rag-wikipedia
Number of documents loaded via DirectoryLoader is 165
Uploaded vectordb to selvi-faiss-vectordbs selvi-s3-rag-wikipedia
```

After the job completes, please ensure that the Vector Store file has been created in your S3 bucket. Here is a screenshot of the Vector Store file for the mini RAG dataset:

You can use this AWS cli command to verify that it was created correctly:
```sh
% aws s3 ls $VECTOR_DB_S3_BUCKET/$VECTOR_DB_S3_FILE
```
```sh
2024-10-30 12:52:52  104804503 selvi-s3-rag-wikipedia
```

## Setup RAG + LLM service

We will now create a Kubernetes Deployment and a Service that will take in the user’s question, interact with the Vector Store to find relevant documents and then query our hosted LLM service to provide an answer. You can download the manifest chat-serveragllm.yaml from here: [chat-serveragllmpluslb.yaml](https://github.com/elotl/GenAI-infra-stack/blob/main/demo/llm.rag.service/chat-serveragllmpluslb.yaml)

```sh
envsubst < chat-serveragllmpluslb.yaml | kubectl apply -f -
```

Please wait for the deployment and Kubernetes LoadBalancer service to become ready and to also obtain an external IP. This can take a few minutes. The command outputs below specifically show the deployment, pod and services associated with the RAQ LLM service.

```sh
# View deployments

% kubectl get deploy
NAME                 	READY   UP-TO-DATE   AVAILABLE   AGE
serveragllm-deployment   1/1 	1        	1       	2m12s
...

# View pods
% kubectl get pods  
NAME                                                  	READY   STATUS	RESTARTS   AGE
serveragllm-deployment-7bcd47c9dc-nqs2s               	1/1 	Running   0      	2m15s

# View services
% kubectl get svc   
NAME                      TYPE           CLUSTER-IP   	EXTERNAL-IP          PORT(S)          AGE
serveragllm-service       LoadBalancer   10.100.211.63	<some-external-IP>   80:32581/TCP     2m19s
```

Please note the IP listed in the EXTERNAL-IP column shown in the output of the `kubectl get svc` command above.


## Query the LLM with RAG
You can use the CURL command or Postman to access the RAG+LLM service endpoint and ask questions about your RAG dataset.

```sh
curl -X GET "http://<some-external-IP>/answer/what%20are%20the%20two%20types%20of%20elephants%20in%20Africa?"

{
"question":"what are the two types of elephants in Africa",
"answer":"The two types of elephants in Africa are the savanna elephant and the forest elephant."
}  
```

# Uninstallation

## 1. Uninstall Model Serve Stack

### A. [MosaicML Open Source Model](https://huggingface.co/mosaicml/mpt-7b-chat)

Uninstall RayService w/vLLM + MosaicML OS Model w/o Ray Autoscaler

```sh
kubectl delete -f ray-service.llm.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.yaml
# After around 5m, Luna will scale down the nodes that were allocated for the RayService
```

Uninstall RayService w/vLLM + MosaicML OS Model w/ Ray Autoscaler

```sh
kubectl delete -f ray-service.llm.autoscale.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.autoscale.yaml
# After around 5m, Luna will scale down the nodes that were allocated for the RayService
```

### B. [Microsoft Open Source Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

Uninstall RayService w/vLLM + Microsoft OS Model w/o Ray Autoscaler

```sh
kubectl delete -f ray-service.llm.Phi-3-mini-4k-instruct.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.Phi-3-mini-4k-instruct.yaml
# After around 5m, Luna will scale down the nodes that were allocated for the RayService
```

## 2. Uninstall RayService w/vLLM + Microsoft OS Model w/ Ray Autoscaler
```sh
kubectl delete -f ray-service.llm.Phi-3-mini-4k-instruct.autoscale.yaml
```
```sh
# From https://github.com/elotl/skyray/blob/main/luna-llm-serve/ray-service.llm.Phi-3-mini-4k-instruct.autoscale.yaml
# After around 5m, Luna will scale down the nodes that were allocated for the RayService
```

## 3. Uninstall Infrastructure Tools

### Uninstall KubeRay
```sh
helm uninstall kuberay-operator
```

### Uninstall Luna

Uninstall Luna as per Installation/Cleanup for cloud K8s type in the [Luna Docs](https://docs.elotl.co/luna/intro/).

# Potential Development Areas
* Replace one shot question/answer with interactive chat, also provide browser interface
* Provide straightforward mechanism to update models
