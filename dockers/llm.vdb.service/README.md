# Create Vector Databas

## Run locally
Create a venv and install requirements.txt

Locally vector db creation can be run in two modes: local dir or s3. 

Please see template files
```shell
.env_local_template
.env_s3_template
```

To run the app You can either export all required env vars or prepare an .env file and run 
```shell
uv run createvectordb.py 
or
python createvectordb.py 
```

Or You can pass the file:
```shell
uv run createvectordb.py --env_file backup-.env
```

## Run tests
Also install in your venv requirements-dev.tx and call
```shell
uv run pytest
```

## Run in k8s
Using pydantic settings introduces one impediment. Before running
```shell
envsubst < createvdb.yaml | kubectl apply -f -
```
We must make sure all env variables are exported. If for ex EMBEDDING_CHUNK_SIZE is not set in our terminal, 
envsubst will put an empty string there and pydantic settings will complain that they can't change 
empty setting to integer.