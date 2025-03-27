# Training the question classification model

## Choose a model version

```bash
export CLASSIFICATION_MODEL_VERSION=v1.2
```

## Create a question training dataset


Choose a name in this pattern:

```bash
syntheticquestions_v1.2.csv
```

## Run the training

```bash
selvik@Selvis-MacBook-Pro train % source .venv/bin/activate           
```

```bash
(train) selvik@Selvis-MacBook-Pro train % uv pip install -r requirements.txt  
Audited 3 packages in 18ms
```


```bash
(train) selvik@Selvis-MacBook-Pro train % uv run question_classification_models.py                  
Training question classification model version: v1.2
Read data from synthetic data file: syntheticquestions_v1.2.csv 

ID to category Dict: {0: 'aggregation', 1: 'pointed'}
Each of the 194 questions is represented by 1034 features (TF-IDF score of unigrams and bigrams)
Model training starts at  2025-03-26 15:35:41
Model training completed at  2025-03-26 15:35:41 

Testing with a sample question:  What was the last upgrade issue? 
Predicted Question Type: aggregation
Testing with a sample question:  How can I upgrade PhP on our server? 
Predicted Question Type: pointed
Model and vectorizer saved successfully!
```
