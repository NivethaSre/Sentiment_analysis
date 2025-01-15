# Sentiment Analysis on Tweets

## Project Overview
This project performs sentiment analysis on tweets, classifying them as **positive**, **negative**, or **neutral**. The analysis uses supervised machine learning techniques, specifically the Naive Bayes classifier, to build a robust sentiment model based on the provided datasets.

## Features
- Preprocessing of text data (cleaning, tokenization, and stopword removal).
- Train-test-validation split for model evaluation.
- Vectorization of text using CountVectorizer.
- Classification using Multinomial Naive Bayes.
- Performance evaluation with accuracy and classification reports.

## Prerequisites
To run the project, ensure the following tools and libraries are installed:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk

To install the required libraries, run:
```bash
pip install pandas scikit-learn nltk
```

## Dataset
The project uses two datasets:
1. `twitter_training.csv`: The training dataset.
2. `twitter_validation.csv`: The validation dataset.

### Dataset Structure
Ensure the datasets have the following structure:
- `label`: Sentiment of the tweet (positive, negative, neutral).
- `text`: The content of the tweet.

If the datasets do not have headers, the script dynamically assigns column names.

## Project Files
- `sentiment.py`: The main script containing the sentiment analysis implementation.
- `twitter_training.csv`: Training dataset.
- `twitter_validation.csv`: Validation dataset.

## Steps to Run the Project

1. Clone the repository or download the project files.
2. Place the datasets (`twitter_training.csv` and `twitter_validation.csv`) in the same directory as the script.
3. Run the script:
   ```bash
   python sentiment.py
   ```
4. View the output, which includes:
   - Training and validation process.
   - Model accuracy on the validation and test datasets.
   - Classification reports with detailed metrics.

## Script Explanation

### Loading the Data
The script dynamically loads and inspects the datasets, adjusting column names as needed:
```python
train_data = load_and_inspect(train_file, "Training Data")
validation_data = load_and_inspect(validation_file, "Validation Data")
```

### Preprocessing
The `preprocess_text` function cleans and tokenizes text, removing URLs, special characters, and stopwords:
```python
def preprocess_text(text):
    # Cleaning and tokenization logic here
```

### Model Training and Evaluation
1. Text data is vectorized using `CountVectorizer`.
2. A Naive Bayes classifier is trained on the training dataset.
3. Validation and test sets are evaluated for accuracy and performance metrics:
```python
classifier.fit(X_train_vectorized, y_train)
y_val_pred = classifier.predict(X_val_vectorized)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
```

## Outputs
The script outputs:
1. Validation Accuracy.
2. Test Accuracy.
3. Detailed Classification Reports.

## Example Output
```
Validation Accuracy: 0.85
Classification Report (Validation):
               precision    recall  f1-score   support

    negative       0.82      0.84      0.83       200
    neutral        0.88      0.81      0.84       250
    positive       0.85      0.89      0.87       300

   accuracy                           0.85       750
  macro avg       0.85      0.85      0.85       750
weighted avg       0.85      0.85      0.85       750

Test Accuracy: 0.84
Classification Report (Test):
...
```

## Customization
You can modify the following:
- **Preprocessing**: Customize `preprocess_text` to handle specific text patterns.
- **Classifier**: Replace Naive Bayes with another model (e.g., SVM, Logistic Regression).
- **Vectorizer**: Experiment with `TfidfVectorizer` for better feature extraction.

## Troubleshooting
1. **Error: Column Mismatch**
   - Ensure the dataset contains the required `label` and `text` columns.
   - Check dataset structure and adjust column names if necessary.

2. **Library Import Errors**
   - Ensure all required libraries are installed using `pip`.

3. **Low Accuracy**
   - Investigate preprocessing steps and dataset quality.
   - Tune hyperparameters or try a different classifier.


