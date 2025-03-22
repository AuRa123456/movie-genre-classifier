import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load training data
def load_data(filename, is_train=True):
    try:
        data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(' ::: ')
                if is_train and len(parts) == 4:
                    data.append(parts)
                elif not is_train and len(parts) == 3:
                    data.append(parts)

        if not data:
            raise ValueError("File is empty or incorrectly formatted.")

        if is_train:
            df = pd.DataFrame(data, columns=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
        else:
            df = pd.DataFrame(data, columns=['ID', 'TITLE', 'DESCRIPTION'])

        # Fill missing values
        df.fillna("Unknown", inplace=True)
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit()
    except ValueError as e:
        print(f"Data Error: {e}")
        exit()

# Load datasets
train_df = load_data('datasets/train_data.txt', is_train=True)
test_df = load_data('datasets/test_data.txt', is_train=False)

# Check class distribution
print("Class Distribution:\n", train_df["GENRE"].value_counts())

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=7000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train_df['DESCRIPTION'])
y = train_df['GENRE']

# Encode genre labels into numerical values
genre_to_index = {genre: i for i, genre in enumerate(y.unique())}
index_to_genre = {i: genre for genre, i in genre_to_index.items()}
y = y.map(genre_to_index)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
classifier = XGBClassifier(
    n_estimators=100, # Number of trees (higher = more accuracy but slower)
    max_depth=6, # Depth of trees (controls complexity)
    learning_rate=0.1, # Step size (higher = faster but risk of overfitting)
    objective="multi:softmax",
    num_class=len(genre_to_index),
    tree_method="hist",                     
    verbosity=1
)

classifier.fit(X_train, y_train)

# Evaluate model
val_predictions = classifier.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
report = classification_report(y_val, val_predictions, zero_division=0, output_dict=True)

# Print classification report
print("Validation Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_val, val_predictions, zero_division=0))

# Predict on test data
X_test = vectorizer.transform(test_df['DESCRIPTION'])
test_predictions = classifier.predict(X_test)
test_df['PREDICTED_GENRE'] = [index_to_genre[pred] for pred in test_predictions]

# Save results
test_df.to_csv('predicted_test_data.csv', index=False)
print("Predictions saved to predicted_test_data.csv")

# Final summary
print("\n=== Model Performance Summary ===")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Precision: {report['weighted avg']['precision']:.4f}")
print(f"Recall: {report['weighted avg']['recall']:.4f}")
print(f"F1 Score: {report['weighted avg']['f1-score']:.4f}")
