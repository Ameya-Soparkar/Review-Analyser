from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)




root = Tk()
root.title('Review Analyser')
root.geometry("500x500")

def good_bad():
    line = entry.get()
    line = re.sub('[^a-zA-Z]', ' ', line) #kept all letters

    line = line.lower() #turned everything to lower case

    line = line.split() #all words split

    ps = PorterStemmer()

    all_stopwords_line = stopwords.words('english')
    all_stopwords_line.remove('not')

    line = [ps.stem(word) for word in line if not word in set(all_stopwords_line)]
    line=[' '.join(line)]
    #print(line)
    sample = cv.transform(line).toarray()
    #print(sample)
    predicted = classifier.predict(sample)
    if predicted == 0:
        return 'The Customer didnt like your service'
    else:
        return 'The Customer liked your services'

def deletetext():
    label.destroy()

def clickk():
    global label
    final = good_bad() 
    label = Label(root, text=final)
    label.grid(row=3,column=0)


line_rev=StringVar()
entry = Entry(root, width=50, textvariable=line_rev)
entry.grid(row=0, column=0)
#entry.insert(0, "Enter your review")

button = Button(root, text='Submit', command=clickk)
button.grid(row=1,column=0)

button2=Button(root, text='Delete', command=deletetext)
button2.grid(row=2,column=0)

root.mainloop()



