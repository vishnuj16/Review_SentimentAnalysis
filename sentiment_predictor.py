import numpy as np
import pandas as pd
import re
import nltk
import pickle
import joblib


#Getting Datasets
dataset = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv', delimiter='\t', quoting=3)

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

# data transformation
from sklearn.feature_extraction.text import CountVectorizer
cvFile = 'Bow_Sentiment_Model.pkl'
cv = pickle.load(open(cvFile, 'rb'))

X_fresh = cv.transform(corpus).toarray()
print(X_fresh.shape)

classifier = joblib.load('Classifier_Sentiment_Model')

y_pred = classifier.predict(X_fresh)
print(y_pred)

dataset['predicted_label'] = y_pred.tolist()
dataset.head()

dataset.to_csv('Predicted_Sentiments.tsv', sep='\t', encoding='UTF-8', index=False)


