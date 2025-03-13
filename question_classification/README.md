# ML-based classification for the Question Router

## Model training 

Ensure that you have a CSV with questions to be used for training.

```bash
% cd train
% uv venv                                 
% source .venv/bin/activate                                         
% uv pip install -r requirements.txt   
```

```bash
% echo $CLASSIFICATION_MODEL_VERSION                        
v1.1
```

```bash
% uv run question_classification_models.py
```

```bash
Training question classification model version: v1.1
Read data from synthetic data file: syntheticquestions_v1.1.csv 

ID to category Dict: {0: 'aggregation', 1: 'pointed'}
Each of the 194 questions is represented by 1034 features (TF-IDF score of unigrams and bigrams)
Model training starts at  2025-03-13 11:57:24
Model training completed at  2025-03-13 11:57:24 

Testing with a sample question:  What was the last upgrade issue? 
Predicted Question Type: aggregation
Testing with a sample question:  How can I upgrade PhP on our server? 
Predicted Question Type: pointed

Model and vectorizer saved successfully!
```

