import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def predict_question_type(question, model, tfidf, id_to_category):
    # Transform the input question into TF-IDF feature representation
    question_tfidf = tfidf.transform([question]).toarray()

    # Predict the category ID
    predicted_category_id = model.predict(question_tfidf)[0]

    # Convert category ID back to label
    predicted_category = id_to_category[predicted_category_id]

    return predicted_category

# Save the trained model
def save_models(rf_model, tfidf):
    joblib.dump(rf_model, 'random_forest_model.pkl')

    # Save the TF-IDF vectorizer as well (since it's needed for transforming new data)
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    print("Model and vectorizer saved successfully!")

# loading data, replace with questions specific to your dataset
df = pd.read_csv('syntheticquestions.csv')

# Create a new dataframe with two columns
df1 = df[['question', 'question_type']].copy()
df1.head(3).T
df2=df1

# map categories to numbers
# Create a new column 'category_id' with encoded categories
df2['category_id'] = df2['question_type'].factorize()[0]
category_id_df = df2[['question_type', 'category_id']].drop_duplicates()

# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'question_type']].values)

print("ID to category Dict:", id_to_category)

# New dataframe
df2.head()

# find features and labels
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1,
                        ngram_range=(1, 2),
                        stop_words='english')

# transform each question into a vector
features = tfidf.fit_transform(df2.question).toarray()
labels = df2.category_id
print("Each of the %d questions is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

print("Model training starts...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
rf_model.fit(features, labels)

sample_question = "What was the last upgrade issue?"
predicted_category = predict_question_type(sample_question, rf_model, tfidf, id_to_category)
print("Testing with a sample question: ", sample_question, "\nPredicted Question Type:", predicted_category)

save_models(rf_model, tfidf)