# Classifier-models-on-Steam-Reviews
Steam Reviews Sentiment Analysis

This project explores sentiment analysis on a dataset of Steam reviews. The goal is to classify reviews as either Positive or Negative based on the text content of the user reviews using various Machine Learning algorithms.

📌 Project Overview

The notebook steamReviews_sentimentAnalysis.ipynb takes raw review data, processes the natural language text, and compares the performance of several supervised learning models. It covers the entire pipeline from data ingestion to model evaluation.

Key Features

Exploratory Data Analysis (EDA): Visualizing the distribution of review lengths, sentiment balance, and word frequency (WordClouds).

Text Preprocessing: Cleaning text data by removing punctuation, stopwords, and special characters to prepare it for vectorization.

Feature Extraction: Converting text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) and Count Vectorization.

Model Comparison: Training and evaluating multiple classifiers to find the best performing model.

🛠️ Technologies Used

Python 3.x

Pandas: Data manipulation and analysis.

NumPy: Numerical computing.

Matplotlib & Seaborn: Data visualization.

NLTK (Natural Language Toolkit): Text processing (stopwords, tokenization).

Scikit-Learn: Machine learning models and evaluation metrics.

WordCloud: Visualizing frequent words.

⚙️ Installation & Setup

Clone the repository:

git clone [https://github.com/chirayuxd/classifier-models-on-steam-reviews.git](https://github.com/chirayuxd/classifier-models-on-steam-reviews.git)
cd classifier-models-on-steam-reviews


Install dependencies:
Ensure you have Python installed, then run:

pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud


Download NLTK data (if required):
Open a python shell and run:

import nltk
nltk.download('stopwords')
nltk.download('punkt')


🚀 Notebook Workflow

The analysis proceeds through the following stages:

Data Loading: Importing the Steam Reviews dataset.

Data Visualization:

Checking the balance of target labels (Positive/Negative).

Analyzing review text length distributions.

Generating WordClouds to see prominent words in positive vs. negative reviews.

Data Preprocessing:

Lowercasing text.

Removing punctuation and special characters.

Removing English stop words (e.g., "the", "is", "at").

Model Training:
The following classifiers are implemented and compared:

Multinomial Naive Bayes: Excellent baseline for text classification.

Logistic Regression: Probabilistic linear classifier.

K-Nearest Neighbors (KNN): Distance-based classification.

Support Vector Machine (SVM): (If included in final iterations) for high-dimensional margin separation.

Decision Tree / Random Forest: Tree-based ensemble methods.

Evaluation:
Models are evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

📊 Results

Naive Bayes typically performs very well on text data with high speed.

Logistic Regression often provides the best balance between interpretability and accuracy for binary sentiment tasks.

Detailed accuracy scores and confusion matrices can be found at the end of the notebook execution.

🤝 Contributing

Contributions are welcome! If you'd like to improve the preprocessing steps or add deep learning models (like LSTMs or Transformers), feel free to fork the repository and submit a pull request.

📜 License

This project is open-source and available under the MIT License.
