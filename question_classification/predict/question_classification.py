import os
import joblib

def predict_question_type(question, model, tfidf, id_to_category):
    
    # Transform the input question into TF-IDF feature representation
    question_tfidf = tfidf.transform([question]).toarray()

    # Predict the category ID
    predicted_category_id = model.predict(question_tfidf)[0]

    # Convert category ID back to label
    predicted_category = id_to_category[predicted_category_id]

    return predicted_category

def load_models():
    # Load the saved model
    rf_model_loaded = joblib.load('./models/random_forest_model.pkl')

    # Load the saved TF-IDF vectorizer
    tfidf_loaded = joblib.load('./models/tfidf_vectorizer.pkl')

    print("Model and vectorizer loaded successfully!")
    return rf_model_loaded, tfidf_loaded

sample_question = "How many tickets are there?"
#sample_question = "What was the last upgrade issue?"
rf_model_loaded, tfidf_loaded = load_models()
id_to_category = {0: 'aggregation', 1: 'pointed'}
predicted_category = predict_question_type(sample_question, rf_model_loaded, tfidf_loaded, id_to_category)

print("Testing with a sample question: ", sample_question, "\nPredicted Question Type:", predicted_category)
