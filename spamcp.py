#import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

model = pickle.load(open('model.pkl','rb'))
message = pickle.load(open('vectorizer.pkl','rb'))

input_sms = "Free entry in 2 a wkly comp to win FA Cup finalr"

# 1. preprocess
#transformed_sms = transform_text(input_sms)
transformed_sms = transform_text(input_sms)

# 2. vectorize
vector_input = message.transform([transformed_sms])
# 3. predict
result = model.predict(vector_input)[0]
# 4. Display
if result == 1:
    print("Spam")
else:
    print("Not Spam")