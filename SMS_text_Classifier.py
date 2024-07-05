# This is SMS_text_Classifier for determining whether a message is spam or ham
#Importing the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV,KFold


# %%
# get data files
#!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
#!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# %% [markdown]
# Read data from the 'tsv'file using pandas and specifying the seperator as tab

# %%
train_data = pd.read_csv(train_file_path,sep='\t',names=['Label','message'])
test_data  = pd.read_csv(test_file_path,sep='\t',names=['Label','message'])

# %%
#Shape of both train and Test data
print("Test:",test_data.shape,'Train:',train_data.shape)

# %%
train_data['Label'] = train_data['Label'].map({'ham':0,'spam':1}).astype(int)
test_data['Label'] = test_data['Label'].map({'ham':0,'spam':1}).astype(int)
X_train = train_data['message']
X_test = test_data['message']

# %% [markdown]
# Using TfidVectorizer for vectorization(text_to_numeric form on the avaliable sms messages)

# %%
Vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
X_train_V = Vectorizer.fit_transform(X_train).toarray()
X_test_V = Vectorizer.transform(X_test).toarray()

# %%
Y_train = train_data.pop('Label')
Y_test = test_data.pop('Label')

# %% [markdown]
# Performing Crossing Validation  using the GridSearchCV in order to determine the best paramater for my Model

# %%
#Using GaussianNaivebayes algorithm 
kf = KFold(n_splits=5,shuffle=True,random_state=42)
params = {'var_smoothing':[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]}
model=GaussianNB()
model_cv = GridSearchCV(model,param_grid=params,cv=kf)
model_cv.fit(X_train_V,Y_train)
print("Best Paramater:",model_cv.best_params_,'Best score:',model_cv.best_params_)

# %%
y_predict = model_cv.predict(X_test_V)
print(classification_report(Y_test,y_predict))
print(confusion_matrix(Y_test,y_predict))

# %%
# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  prediction =[]
  #convert text to numeric
  Pred = Vectorizer.transform([pred_text]).toarray()
  predicts = model_cv.predict_proba(Pred)[0][1]
  print(predicts)
  if predicts < 0.5:
    prediction.append(predicts)
    prediction.append('ham')
  else:
    prediction.append(predicts)
    prediction.append('spam')
  return (prediction)

pred_text = "our new mobile video service is live. just install on your phone to start watching."

prediction = predict_message(pred_text)
print(prediction)

# %%
# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()



