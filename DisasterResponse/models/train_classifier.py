import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import joblib


def load_data(database_filepath):
    """
    function to load the data from SQL Lite database
    parameter will be the path of the database 
    the SQL table will be converted to pandas data frame 
    which will then be processed to dependent and independent 
    variables    
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    cat_names = list(df.columns[4:])
    return X, Y, cat_names


def tokenize(text):
    """
    function to perform text processing by scanning for the URL,
    tokenize text, lemmatize, normalize case and removing trailing and white spaces
    parameter would be passing the text
    
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # look for the URL
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "URLSAMPLE")
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build modeling using SKLearn's pipeline
    also utilize grid search 
    """

    model = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=6)))
    ])

    parameters = {'clf__estimator__max_features': ['sqrt', 0.5],
                  'clf__estimator__n_estimators': [80, 100]}

    cv = GridSearchCV(estimator=model, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    result_set = []
    for i in range(len(y_test.columns)):
        result_set.append([f1_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                           precision_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                           recall_score(y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    result_set = pd.DataFrame(result_set, columns=['f1 score', 'precision', 'recall'],
                              index=y_test.columns)
    print(result_set)
    return result_set


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, cat_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
