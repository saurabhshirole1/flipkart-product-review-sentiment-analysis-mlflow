# 🛒 Flipkart Review Sentiment Analysis using mlflow  
### NLP · Machine Learning · MLflow · MLOps

![Project Banner](./banner.png)

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#application-preview">Application Preview</a>
- <a href="#business-objective">Business Objective</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-preparation">Data Preparation</a>
- <a href="#model--approach">Model & Approach</a>
- <a href="#mlflow--experiment-tracking--mlops">MLflow – Experiment Tracking & MLOps</a>
- <a href="#key-insights">Key Insights</a>
- <a href="#business-impact">Business Impact</a>
- <a href="#how-to-run-the-project">How to Run the Project</a>
- <a href="#author--contact">Author & Contact</a>
- <a href="#acknowledgment">Acknowledgment</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

The **Flipkart Review Sentiment Analysis Project** is an end-to-end **Natural Language Processing (NLP)** and **Machine Learning** solution that analyzes customer reviews and classifies them into **Positive** or **Negative** sentiments.

This project integrates **MLflow** for **experiment tracking, model management, and reproducibility**, following real-world **MLOps best practices**.

---


<h2><a class="anchor" id="business-objective"></a>Business Objective</h2>

- Analyze large volumes of Flipkart product reviews  
- Automatically classify reviews as **Positive** or **Negative**  
- Reduce manual effort in customer feedback analysis  
- Enable data-driven decision making for e-commerce platforms  

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

- **Dataset Name:** Flipkart Product Reviews Dataset  
- **Records:** 8,508 reviews  
- **Columns:** Review Text, Sentiment Label, Product Metadata  
- **Source:** Public e-commerce review dataset  
- **Target Variable:** Sentiment (Positive / Negative)

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorizer (Unigrams + Bigrams)  
- MLflow – Experiment Tracking & Model Registry  
- Pickle – Model persistence  
- Streamlit – Web application  

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```


Sentiment analysis project/
│
├── app/
│   └── app.py                  # Streamlit web app
│
├── data/
│   └── flipkart_reviews.csv    # Dataset
|
├── images
|   └── app.png
|
├── model_building/
│   └── model_building.py       # Model training & evaluation
│
├── notebook/
|   └── Sentiment Analysis (EDA & Data Preprocessing).ipynb
|
├── pkl/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

````

---

<h2><a class="anchor" id="data-preparation"></a>Data Preparation</h2>

The following NLP preprocessing steps were applied:

- Removal of missing values  
- Lowercasing text  
- Removal of special characters & punctuation  
- Tokenization  
- Stopword removal  
- Lemmatization  

These steps significantly improved model performance.

---

<h2><a class="anchor" id="model--approach"></a>Model & Approach</h2>

**Feature Engineering**
- TF-IDF Vectorization  
- `max_features = 5000`  
- `ngram_range = (1, 2)`  

**Models Trained & Compared**
- Logistic Regression  
- Naive Bayes  
- Linear Support Vector Machine (SVM)  
- Random Forest  

**Evaluation Metric**
- F1 Score  

**Final Model**
- **Linear SVM** (Best performing model)

---

<h2><a class="anchor" id="mlflow--experiment-tracking--mlops"></a>MLflow – Experiment Tracking & MLOps</h2>

MLflow was integrated to ensure **reproducibility, experiment tracking, and model governance**.

**MLflow Capabilities Used**
- Experiment tracking for multiple models  
- Custom run names for each algorithm  
- Logging parameters and evaluation metrics  
- Model artifact storage  
- Metric & hyperparameter comparison plots  
- Model registration and versioning  

**Model Registry**
- Registered Model Name: **FlipkartSentimentModel**  
- Tagged for lifecycle management and version control  

---

<h2><a class="anchor" id="key-insights"></a>Key Insights</h2>

1. Linear SVM outperformed other models in sentiment classification  
2. TF-IDF with bigrams improved contextual understanding  
3. Proper NLP preprocessing boosted F1 score significantly  
4. MLflow simplified experiment comparison and model selection  

---

<h2><a class="anchor" id="business-impact"></a>Business Impact</h2>

This system enables organizations to:

- Analyze customer sentiment at scale  
- Improve products using sentiment trends  
- Enhance customer satisfaction  
- Save time and cost compared to manual analysis  
- Maintain reproducible and auditable ML pipelines  

---

<h2><a class="anchor" id="how-to-run-the-project"></a>How to Run the Project</h2>

1. **Clone the repository**
```bash
git clone https://github.com/your-username/flipkart-review-sentiment-analysis.git
cd flipkart-review-sentiment-analysis
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train model with MLflow**

```bash
python model_building/train_with_mlflow.py
```

4. **Launch MLflow UI**

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

5. **(Optional) Run Streamlit App**

```bash
streamlit run app/app.py
```

---

<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

**Nikhil Borade**
Aspiring Data Scientist | Machine Learning | NLP | MLOps

* GitHub: [https://github.com/nikhilborade0412](https://github.com/nikhilborade0412)
* LinkedIn: [https://www.linkedin.com/in/nikhilborade](https://www.linkedin.com/in/nikhilborade)

---

<h2><a class="anchor" id="acknowledgment"></a>Acknowledgment</h2>

Special thanks to **Innomatics Research Labs** for providing hands-on, industry-oriented training and continuous mentorship throughout this project.

```
