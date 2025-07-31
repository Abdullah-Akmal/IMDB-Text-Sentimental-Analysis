# IMDB Text Sentiment Analysis

## Problem Statement
Analyzing thousands of movie reviews to understand audience sentiment is time-consuming and subjective. This project automates sentiment classification using Natural Language Processing (NLP) and machine learning, enabling efficient and accurate categorization of IMDB reviews as positive or negative


---

## Tools & Technologies Used

- **Programming Language**: Python  
- **Libraries**:  
  - `pandas`, `numpy` for data manipulation  
  - `re`, `nltk` for text cleaning, tokenization, stopword removal, and lemmatization  
  - `scikit-learn` for vectorization (TF-IDF, CountVectorizer), model training (Logistic Regression, Naive Bayes), and evaluation  
  - `matplotlib`, `seaborn` for data visualization  
  - `tkinter` for building a simple GUI  
  **[Dataset Source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**

---

## Process Breakdown

### 1. Data Exploration
- Loaded the IMDB dataset with 50,000 labeled reviews (positive/negative)
- Verified class balance and inspected sample reviews

### 2. Text Preprocessing
- Removed HTML tags, special characters, and converted text to lowercase
- Applied tokenization, stopword removal, and lemmatization using NLTK
- Converted cleaned text into numerical vectors using **TF-IDF** and **CountVectorizer** (top 10,000 features)

### 3. Model Training & Evaluation
- Split data into 80% training and 20% testing sets using stratified sampling
- Trained and compared two models:
  - Logistic Regression (optimized with 1,000 iterations)
  - Multinomial Naive Bayes (probabilistic baseline)
- Evaluated models using Accuracy, Precision, Recall, and F1-Score
- Generated classification reports and confusion matrices

### 4. Visualization & Deployment
- Visualized word frequencies and model performance metrics
- Developed a **Tkinter GUI** that allows users to:
  - Input a review
  - Select a model
  - Receive real-time sentiment predictions

---

## Model Performance Comparison

| Model                   | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.8941   | 0.8861    | 0.9044 | 0.8952   |
| Multinomial Naive Bayes| 0.8584   | 0.8568    | 0.8606 | 0.8587   |

---

## Project Outcome

- Logistic Regression proved more effective, achieving nearly **89.4% accuracy**, and better capturing sentiment nuances in review text
- The model can scale to analyze large volumes of feedback across domains (e.g., e-commerce, social media)
- The pipeline offers a foundation for businesses to automate customer sentiment analysis and improve content strategies

---

## Key Insight

Logistic Regression outperforms Naive Bayes by better capturing contextual polarity (e.g., "not great" vs. "great"), thanks to its ability to model linear relationships in TF-IDF-transformed data.

---


