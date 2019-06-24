# Builds a model that categorizes wine reviews by the probable type of wine.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import  SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

reviews = pd.read_csv("winemag-data-130k-v2.csv")
reviews.dropna(inplace=True)
varieties = reviews["variety"].unique().tolist()

# Split data into test/training data sets

ignored_words_set = text.ENGLISH_STOP_WORDS.union(varieties)
vectorizer = TfidfVectorizer(stop_words=ignored_words_set)

vectorized_descriptions = vectorizer.fit_transform(reviews['description'].values)

train_features, test_features, train_labels, test_labels = train_test_split(vectorized_descriptions, reviews["variety"])

# Test a few different models and see which one is most accurate
models = [MultinomialNB(), BernoulliNB(), ComplementNB(), LogisticRegression(solver='lbfgs', multi_class='auto'), Perceptron(), SGDClassifier(), DecisionTreeClassifier(), MLPClassifier()]
for model in models:
    model.fit(train_features, train_labels)
    name = model.__class__.__name__
    prediction = model.predict(test_features)
    accuracy = accuracy_score(test_labels, prediction)
    print("Accuracy of ", name, ": ", accuracy)
