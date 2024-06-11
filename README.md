# Fake News Detection

Fake News Detection is a machine learning project aimed at identifying fake news articles from a dataset of 70,000 news articles. The project leverages the Random Forest algorithm to predict the authenticity of news based on various textual and numeric features.

## Features

- **Data Cleaning**: Process the dataset to remove noise and inconsistencies.
- **Feature Engineering**: Create textual and numeric features for model training.
- **Model Training**: Train a Random Forest model on the processed dataset.
- **Model Testing**: Evaluate the model's performance on a test dataset.

## Technology Stack

- **Programming Language**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `nltk`, `re`, `matplotlib`

## Installation

### Prerequisites

- Python 3.7+
- Install required libraries

### Install Required Python Libraries

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for this project consists of 70,000 news articles, labeled as fake or real. The dataset should be structured in a CSV format with the following columns:

- `title`: Title of the news article
- `text`: Full text of the news article
- `label`: Label indicating whether the news is fake (0) or real (1)

## Usage

1. **Run the Fake News Detection script**:
    ```bash
    python fake_news_detection.py
    ```

2. **Script Functionality**:
    - **Data Cleaning**: Preprocess the text data to remove punctuation, stop words, and perform tokenization.
    - **Feature Engineering**: Extract textual and numeric features such as TF-IDF vectors.
    - **Model Training**: Train the Random Forest model on the training dataset.
    - **Model Testing**: Test the model on the test dataset and evaluate its performance.

## Features Implementation

### Data Cleaning

```python
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_data(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    return text

def preprocess_data(df):
    stop_words = set(stopwords.words('english'))
    df['clean_text'] = df['text'].apply(clean_data)
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return df
```

### Feature Engineering

```python
def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    return X
```

### Model Training and Testing

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Classification Report:\n{report}')
```

### Running the Script

```python
if __name__ == "__main__":
    df = pd.read_csv('path/to/dataset.csv')
    df = preprocess_data(df)
    X = extract_features(df)
    y = df['label']
    model, y_test, y_pred = train_model(X, y)
    evaluate_model(y_test, y_pred)
```

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact [dheerajdubey600@gmail.com](mailto:your-email@example.com).

---

This README provides an overview of the Fake News Detection project, including setup instructions, key functionalities, and how to contribute. For further details, please refer to the project source code.
