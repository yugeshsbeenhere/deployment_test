import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
import joblib
import pickle
df = pd.read_csv("NAOP_new.csv", encoding="latin-1")
df.drop(df.iloc[:, 3:], inplace = True, axis = 1) 
df.drop(df.iloc[:, 0:1], inplace = True, axis = 1) 
df['Type'] = df['Type'].map({'A': 0, 'B': 1,'C': 2, 'D': 3})
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
df['Posts']=df['Posts'].apply(lambda x: remove_URL(x))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['Posts']=df['Posts'].apply(lambda x: remove_punct(x))

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text
df['Posts']=df['Posts'].apply(lambda x: remove_numbers(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df['Posts']=df['Posts'].apply(lambda x: remove_html(x))

STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower() 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['Posts']=df['Posts'].apply(lambda x: clean_text(x))

X = df['Posts']
y = df['Type']
cv = CountVectorizer()
X = cv.fit_transform(X) 
pickle.dump(cv,open('transform.pkl', 'wb'))
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X,y)
pickle.dump(sm,open('sample.pkl', 'wb'))
X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.10, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename= 'naop_model.pickle'
pickle.dump(clf,open('naop_model.pkl', 'wb'))
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
