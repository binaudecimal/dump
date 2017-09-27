import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("dtr.csv")

df_x = df["Message"]
df_y = df["Category"]

cv = TfidfVectorizer(min_df=1, stop_words='english')
x_traincv = cv.fit_transform(df_x)
a = x_traincv.toarray()
clf = svm.SVC(gamma=0.001, C=100)

x,y = a, df_y

clf.fit(x,y)
text = "need po ng rescue matanda karamdaman help"
array = cv.transform([text])
print(clf.predict(array.toarray()))
print(cv.inverse_transform(array))
#print("Prediction : ", clf.predict(cv.fit_transform(["Kailangan ko ng tulong, nawawala ang anak ko"])))