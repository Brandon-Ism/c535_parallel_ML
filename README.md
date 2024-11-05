# c535_parallel_ML
This repository includes a Jupyter Notebook for spam detection using a sequential approach. The notebook demonstrates how machine learning models can classify text messages as spam or not spam based on provided datasets and evaluates their performance.

**Project Overview**
Spam detection is a classic problem in natural language processing (NLP) and machine learning. In this project, a sequential (non-parallel) approach is employed to classify messages as spam or non-spam using popular machine learning techniques.
**File Structure**
sequential_spam_detect.ipynb: The primary Jupyter Notebook for this project, where data preprocessing, feature extraction, model training, and evaluation are carried out.
**Setup Instructions**
Clone the Repository:
git clone https://github.com/Brandon-Ism/c535_parallel_ML.git
cd c535_parallel_ML/notebooks_b
**Install Required Libraries:** Make sure you have Python installed (3.7+). Then, install the required packages:
pip install -r requirements.txt
**The following libraries are necessary for running this notebook:**

**pandasFor:** data manipulation and analysis
**numpy:** For numerical operations
**scikit-learn**: For machine learning algorithms and evaluation metrics
**nltk**: For natural language processing, especially text pre-processing
**matplotlib & seaborn**: For data visualization (optional)
**Dataset:** This notebook assumes access to a labeled dataset of text messages for spam classification, commonly in CSV format with columns such as text and label.

Ensure your dataset file is in the same directory or adjust the file path in the notebook.
Notebook Overview
Data Loading:

Reads in the dataset, handling any missing values and verifying the data structure.
Data Preprocessing:

Tokenization, text normalization, stopword removal, and other necessary steps to clean and prepare the text data.
Feature Extraction:

Uses Term Frequency-Inverse Document Frequency (TF-IDF) vectorization or other methods to convert text into numerical features.
Model Training:

Trains a spam classifier model, such as Naive Bayes, Logistic Regression, or Support Vector Machine, based on the selected features.
Model Evaluation:

Provides metrics such as accuracy, precision, recall, and F1-score to evaluate model performance.
Includes a confusion matrix to visualize classification results.
Hyperparameter Tuning (Optional):

Hyperparameters may be tuned to optimize model performance.


**Running the Notebook**
To run this notebook, simply open it in Jupyter Notebook or JupyterLab:
jupyter notebook sequential_spam_detect.ipynb

Run each cell sequentially to execute the full spam detection workflow.

**Future Directions**
Parallel Processing: This notebook uses a sequential approach; a parallelized version could further improve processing time, especially for large datasets.
Advanced Models: Consider experimenting with deep learning models (e.g., LSTM or BERT) for improved accuracy on more complex spam detection tasks.
