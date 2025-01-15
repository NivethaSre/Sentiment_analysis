import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# File Paths
train_file = r"C:\Users\nivet\Downloads\archive (9)\twitter_training.csv"
validation_file = r"C:\Users\nivet\Downloads\archive (9)\twitter_validation.csv"

# Load and Inspect Data
def load_and_inspect(file_path, file_name):
    try:
        # Attempt to load the file
        data = pd.read_csv(file_path)
        print(f"Loaded {file_name} successfully!")
        print("Detected Columns:", list(data.columns))
        print("Sample Data:\n", data.head())
        return data
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

# Load Training and Validation Data
train_data = load_and_inspect(train_file, "Training Data")
validation_data = load_and_inspect(validation_file, "Validation Data")

# Check if train_data and validation_data were loaded correctly
if train_data is None or validation_data is None:
    raise ValueError("Failed to load one or both datasets. Please check the file paths and formats.")

# Detect Key Columns
def detect_columns(data, file_name):
    if 'label' in data.columns and 'text' in data.columns:
        return data[['label', 'text']]
    elif len(data.columns) >= 2:  # Assume first two columns are label and text if names are missing
        data.columns = ['label', 'text'] + list(data.columns[2:])
        print(f"Renamed columns for {file_name}: {list(data.columns)}")
        return data[['label', 'text']]
    else:
        raise ValueError(f"Could not detect appropriate columns in {file_name}. Please verify the file structure.")

train_data = detect_columns(train_data, "Training Data")
validation_data = detect_columns(validation_data, "Validation Data")

# Combine Training and Validation Data
data = pd.concat([train_data, validation_data], ignore_index=True)

# Preprocess Text
def preprocess_text(text):
    # Remove URLs, special characters, and convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()  # Convert to lowercase
    text = word_tokenize(text)  # Tokenize words
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Split Dataset into Train, Validation, and Test Sets
X = data['cleaned_text']
y = data['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

# Vectorize Text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Validate the Model
y_val_pred = classifier.predict(X_val_vectorized)
print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Test the Model
y_test_pred = classifier.predict(X_test_vectorized)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))
