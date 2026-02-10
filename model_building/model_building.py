
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score



# Base Directory (Project Root)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "flipkart_reviews.csv")
PKL_DIR = os.path.join(BASE_DIR, "pkl")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)



# MLflow Configuration

mlflow.set_experiment("Flipkart Sentiment Analysis")



# Load Dataset

df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

X = df["clean_review_text"]
y = df["Sentiment"]



# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# TF-IDF Vectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)



# Model Comparison with MLflow
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("ngram_range", "(1,2)")

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        f1 = f1_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("f1_score", f1)

        # Save model as artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # Tags
        mlflow.set_tag("project", "Flipkart Sentiment Analysis")
        mlflow.set_tag("author", "Nikhil Borade")

        results.append({"Model": name, "F1 Score": f1})



# Model Comparison Result

results_df = pd.DataFrame(results).sort_values(
    by="F1 Score", ascending=False
)

print("\nModel Comparison:")
print(results_df)



# Final Model (Best One)

final_model = LinearSVC()
final_model.fit(X_train_tfidf, y_train)

# Save locally (for Streamlit / deployment)
with open(os.path.join(PKL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(PKL_DIR, "sentiment_model.pkl"), "wb") as f:
    pickle.dump(final_model, f)



# Register Final Model in MLflow

with mlflow.start_run(run_name="Final_Linear_SVM"):

    mlflow.log_param("model", "LinearSVM")
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_metric("final_f1_score", results_df.iloc[0]["F1 Score"])

    mlflow.sklearn.log_model(
        final_model,
        artifact_path="sentiment_model",
        registered_model_name="FlipkartSentimentModel"
    )

    mlflow.set_tag("stage", "production_candidate")

print("\n MLflow tracking completed")
print("\n Model registered successfully")

