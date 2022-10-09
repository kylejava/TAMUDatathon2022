import nltk
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import string
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
from preprocess import *
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")
df = pd.read_csv("./spam.csv", encoding = "ISO-8859-1", engine = "python")
df = df[["v1","v2"]].copy()
df = df.rename(columns={"v1": "class", "v2":"sms"})
name_of_values= df["class"].unique().tolist()
num_of_values = []
num_of_values.append(df[df["class"] == "ham"].shape[0])
num_of_values.append(df[df["class"] == "spam"].shape[0])

plt.bar(name_of_values, num_of_values)
plt.show()
df["sms"]=df["sms"].apply(lambda sentence:preprocess(sentence))
df["sms"]=df["sms"].apply(lambda sentence: sentence.lower())
print(df)
sms_values = df["sms"].tolist()
class_values = df["class"].tolist()
x_train, x_test, y_train, y_test = train_test_split(sms_values, class_values, test_size=0.80)
cv = CountVectorizer()
x = cv.fit_transform(x_train)
SVM = svm.SVC()
SVM.fit(x,y_train)
x_test = cv.transform(x_test)
print("Accuracy: " + str(SVM.score(x_test, y_test)))
