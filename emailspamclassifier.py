


#Spam Classifier based on Naive Bayes
The method is called naive because we're asuming there's no relationships between the words themselves. We're looking at each word in isolation, individually within a message, and basically combining all the probabilities of each word's contribution to being spam or not. A better spam classifier would obviously be looking at the relationships between the words.


import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#import metrics libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import pandas as pd
mails=pd.read_csv('spam.csv',encoding='latin-1')

mails.head()

mails.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],inplace=True)
mails= mails.rename(columns={"v1":"label", "v2":"sms"})

mails.head()

"""#Checking the maximum length of SMS
Number of observations in each label spam and ham
"""

mails.label.value_counts()

#Mail text, mail length, mail is ham/spam label
mails['length']=mails['sms'].apply(len)
mails.head(20)

mails['length'].describe()

"""Now i've found max text with the length of 910 in spam dataset.Lets Locate that text in dataset"""

mails[mails['length']==910]['sms'].iloc[0]

"""#Data Preprocessing
Converting the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0 and the 'spam' value to 1.
"""

mails.loc[:,'label'] = mails.label.map({'ham':0, 'spam':1})

mails.head()

mails.shape

ps=PorterStemmer()
lemma=WordNetLemmatizer()

"""#Bag of words
What we have here in our data set is a large collection of text data (5,572 rows of data). Most ML algorithms rely on numerical data to be fed into them as input, and email/sms messages are usually text heavy.
"""

ps=PorterStemmer()#PorterStemmer object
lemma=WordNetLemmatizer()#Here using lemmatization and defining the object
corpus=[]
for i in range(0,len(mails)):
  review=re.sub('[^a-zA-Z]',' ',mails['sms'][i])
  review=review.lower()#Convert all the strings in the documents set to their lower case.
  review=review.split()
  review=[ps.stem(words)for words in review if not words in stopwords.words('english')]
  review=' '.join(review)#joining it using join() function in py 
  corpus.append(review)#appending the processed string in list corpus

"""#Data preprocessing with CountVectorizer()"""

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

"""#Train Test Splits
Now that we have understood how to deal with the Bag of Words problem we can get back to our dataset and proceed with our analysis. Our first step in this regard would be to split our dataset into a training and testing set so we can test our model later.
"""

xSet = mails['sms'].values
ySet = mails['label'].values

X_train, X_test, y_train, y_test = train_test_split(xSet,ySet,test_size=0.20,random_state=1)

# Fit the training data and then return the matrix
training_data = cv.fit_transform(X_train)

# Transform testing data and return the matrix. 
testing_data = cv.transform(X_test)

"""#Naive Bayes Model
With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.
"""

#create and fit NB model
naive_bayes=MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)

"""
Now that predictions have been made on our test set, we need to check the accuracy of our predictions."""

accuracyScore =accuracy_score(y_test,predictions)
print(accuracyScore)

"""#Evaluating our SMS Spam Detection Model"""

#Precision 
print('Precision score: {}'.format(precision_score(y_test, predictions)))
#Recall
print('Recall score: {}'.format(recall_score(y_test, predictions)))
#F1 score
print('F1 score: {}'.format(f1_score(y_test, predictions)))

"""#Confusion Matrix"""

print('Confusion Matrix: {}'.format(confusion_matrix(y_test, predictions)))
