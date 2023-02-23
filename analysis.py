import numpy as np
import pandas as pd
import re
import nltk
import pickle
import joblib


#Getting Datasets
dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t')
print(dataset.shape)

#Data Cleaning
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
allstopwords = stopwords.words('english')
allstopwords.remove('not')

corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(allstopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
# print(corpus)

# Data transformation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)

bow = cv.fit_transform(corpus).toarray()
hl = dataset.iloc[:, -1].values

bow_path = 'Bow_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, 'wb'))

# dividing dataset into training and test state
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bow, hl, test_size=0.20, random_state=0)

#Model Fitting (Naive Bayes)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
GaussianNB(priors=None, var_smoothing=1e-09)

#Exporting NB Classifier to later use in prediction
joblib.dump(classifier, 'Classifier_Sentiment_Model')

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(accuracy_score(Y_test, y_pred))