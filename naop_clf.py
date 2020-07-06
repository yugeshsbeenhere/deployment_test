import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import  classification_report,accuracy_score

train= pd.read_csv("NAOP_new.csv" ,encoding='latin-1')

train.drop(train.iloc[:, 3:], inplace = True, axis = 1) 
train.drop(train.iloc[:, 0:1], inplace = True, axis = 1) 

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
train['Posts']=train['Posts'].apply(lambda x: remove_URL(x))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
train['Posts']=train['Posts'].apply(lambda x: remove_punct(x))

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text
train['Posts']=train['Posts'].apply(lambda x: remove_numbers(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
train['Posts']=train['Posts'].apply(lambda x: remove_html(x))

STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower() 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
train['Posts']=train['Posts'].apply(lambda x: clean_text(x))

A_class = train[train.Type=='A']
B_class = train[train.Type=='B']
C_class = train[train.Type=='C']
D_class = train[train.Type=='D']

df_minority_oversampled = resample(D_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled])
df_oversampled.Type.value_counts()
df_minority_oversampled = resample(B_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled,df_oversampled])
df_minority_oversampled = resample(C_class,replace=True,n_samples=5697,random_state=123)
df_oversampled = pd.concat([df_minority_oversampled,df_oversampled])
df_oversampled = pd.concat([A_class,df_oversampled])
df_oversampled.Type.value_counts()

df_oversampled['Type'] = df_oversampled['Type'].map({'A': 0, 'B': 1,'C': 2, 'D': 3})
X = df_oversampled['Posts']
y = df_oversampled['Type']
cv = CountVectorizer()
X = cv.fit_transform(X)
pickle.dump(cv,open('transform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
#filename= 'naop_model.pickle'
#pickle.dump(clf,open('naop_model.pkl', 'wb'))
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

import sklearn.neural_network
clf_nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', 
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)
clf_nn.fit(X_train,y_train)
pred_ynn = clf_nn.predict(X_test)
print(accuracy_score(y_test,pred_ynn))
print ('\n clasification report:\n', classification_report(y_test,pred_ynn))

from sklearn.neighbors import KNeighborsClassifier as KNC
neigh = KNC(n_neighbors= 4)
neigh.fit(X_train,y_train)
pred_n3=neigh.predict(X_test)
print(accuracy_score(y_test,pred_n3))
print ('\n clasification report:\n', classification_report(y_test,pred_n3))

neigh2 = KNC(n_neighbors= 5)
neigh2.fit(X_train,y_train)
pred_n5=neigh.predict(X_test)
print(accuracy_score(y_test,pred_n5))
print ('\n clasification report:\n', classification_report(y_test,pred_n5))

neigh = KNC(n_neighbors= 6)
neigh.fit(X_train,y_train)
pred_n6=neigh.predict(X_test)
print(accuracy_score(y_test,pred_n6))
print ('\n clasification report:\n', classification_report(y_test,pred_n6))


neigh = KNC(n_neighbors= 3)
neigh.fit(X_train,y_train)
pred_n6=neigh.predict(X_test)
print(accuracy_score(y_test,pred_n6))
print ('\n clasification report:\n', classification_report(y_test,pred_n6))

from sklearn.tree import DecisionTreeClassifier 
clf3 = DecisionTreeClassifier()
clf3.fit(X_train,y_train)
pred_ydt = clf3.predict(X_test)
print(accuracy_score(y_test,pred_ydt))
print ('\n clasification report:\n', classification_report(y_test,pred_ydt))

from sklearn.svm import SVC
clf4 = SVC()
clf4.fit(X_train,y_train)
#from sklearn.metrics import accuracy_score
pred_ysvc = clf4.predict(X_test)
print(accuracy_score(y_test,pred_ysvc))
print ('\n clasification report:\n', classification_report(y_test,pred_ysvc))

from sklearn.linear_model import LogisticRegression 
clf5 = LogisticRegression()
clf5.fit(X_train,y_train)
pred_lr=clf5.pred(X_test)
print(accuracy_score(y_test,pred_lr))
print ('\n clasification report:\n', classification_report(y_test,pred_lr))
