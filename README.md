**Sequential Spam Detection Notebook**
This repository contains a Jupyter Notebook for spam detection using a sequential approach. The notebook demonstrates text classification to distinguish spam from non-spam messages and evaluates model performance using machine learning techniques.
Project Overview
Spam detection is an essential application in natural language processing (NLP) and machine learning. This project leverages a sequential, non-parallel approach to classify text messages as spam or not spam, using established ML models and evaluation metrics.
**File Structure**
sequential_spam_detect.ipynb: The main Jupyter Notebook for the project, which includes:
**Data loading and preprocessing**
 Feature extraction
	Model training and evaluation
 
**Setup Instructions**
1. Clone the Repository
bash
Copy code
git clone https://github.com/Brandon-Ism/c535_parallel_ML.git
cd c535_parallel_ML/notebooks_b
2. Install Required Libraries
Ensure Python 3.7+ is installed, then install the necessary libraries:
bash
Copy code
pip install -r requirements.txt
**Main Libraries**
•	pandas – Data manipulation and analysis
•	numpy – Numerical operations
•	scikit-learn – Machine learning algorithms and evaluation metrics
•	nltk – Natural language processing and text preprocessing
•	matplotlib & seaborn (optional) – Data visualization
3. Dataset
This notebook requires a labeled dataset of text messages, typically in CSV format, with columns such as text and label. Place the dataset file in the same directory or adjust the path within the notebook.

**Notebook Overview**
1.	Data Loading
Loads the dataset, handles missing values, and verifies the structure.
2.	Data Preprocessing
Prepares the text data by tokenizing, normalizing, and removing stopwords.
3.	Feature Extraction
Converts text to numerical features, commonly using TF-IDF vectorization.
4.	Model Training
Trains a spam classifier (e.g., Naive Bayes, Logistic Regression, or SVM) based on extracted features.
5.	Model Evaluation
Evaluates model performance with metrics like accuracy, precision, recall, and F1-score. Also includes a confusion matrix for visual insights.
6.	Hyperparameter Tuning (Optional)
Optimizes the model for improved performance on the dataset.
Running the Notebook
Open the notebook in Jupyter Notebook or JupyterLab and run each cell sequentially:
bash
Copy code
jupyter notebook sequential_spam_detect.ipynb

**Future Directions**
•	Parallel Processing: A parallelized version may significantly reduce processing time for larger datasets.
•	Advanced Models: Experimenting with deep learning models (e.g., LSTM or BERT) could improve accuracy, especially on complex spam detection tasks.
