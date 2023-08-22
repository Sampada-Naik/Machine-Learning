import numpy as np 
import pandas as pd 
import re
import nltk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



nltk.download('stopwords') #list of stopwords 
nltk.download('punkt')#Punkt Sentence Tokenizer. This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences
nltk.download('wordnet') #Wordnet is an NLTK corpus reader, a lexical database for English. It can be used to find the meaning of words, synonym or antonym.



data=pd.read_csv('train.csv',encoding="ISO-8859-1")


print(data.head())
print(data.shape)
data=data.drop_duplicates()
print(data.shape)
print(data.isnull().sum())

data=data.dropna()

print(data.isnull().sum())




#plotting graph
data["Sentiment"].value_counts().plot(kind="bar", color=["salmon","lightblue"])
plt.xlabel("1 = Positive Tweet, 0 = Negative Tweet")
plt.title("Twitter Sentimental Analysis")
plt.show()






#Cleaning and preprocessing
#Remove punctuations from the String  
s = "!</> hello please$$ </>^!!!%%&&%$@@@attend^^^&&!& </>*@# the&&\ @@@class##%^^&!@# %%$"
s = re.sub(r'[^\w\s]','',s)
print(s)



#Tokenization
k=nltk.word_tokenize("Hello how are you")
print(k)




# StopWords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)



sentence = "Covid-19 pandemic has impacted many countries and what it did to economy is very stressful"

words = nltk.word_tokenize(sentence)
words = [w for w in words if w not in stop_words]

print(words)




#Lemmatization

from nltk.stem import WordNetLemmatizer  # It helps in returning the base or dictionary form of a word known as the lemma 
lemmatizer=WordNetLemmatizer()


input_str="been had done languages cities mice"



#Tokenize the sentence
input_str=nltk.word_tokenize(input_str)

#Lemmatize each word
for word in input_str:
    print(lemmatizer.lemmatize(word))






for index,row in data.iterrows():
    
    filter_sentence = ''

    
    sentence = row['SentimentText']

    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning

    words = nltk.word_tokenize(sentence) #tokenization

    words = [w for w in words if not w in stop_words]  #stopwords removal
    
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    #print(filter_sentence)
    data.loc[index,'SentimentText'] = filter_sentence



print(data.head())








X = data['SentimentText']
y= data['Sentiment']
print(X)
print(y)







#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)




print('X_train :', len(X_train))
print('X_test  :', len(X_test))
print('y_train :', len(y_train))
print('y_test  :', len(y_test))






from sklearn.feature_extraction.text import TfidfTransformer  #With Tfidftransformer you will systematically compute word counts using CountVectorizer and then compute the Inverse Document Frequency (IDF) values and only then compute the Tf-idf scores. ... Under the hood, it computes the word counts, IDF values, and Tf-idf scores all using the same dataset
from sklearn.feature_extraction.text import CountVectorizer




from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier
clf1= MultinomialNB()



from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression()



from xgboost import XGBClassifier
clf3 = XGBClassifier()



from sklearn.pipeline import Pipeline



model1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', clf1),
])


model2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', clf2),
])

model3 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', clf3),
])


#training
model1.fit(X_train, y_train)

model2.fit(X_train, y_train)

model3.fit(X_train, y_train)

#testing
predictions1 = model1.predict(X_test)

predictions2 = model2.predict(X_test)

predictions3 = model3.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions1 )

print("Accuracy of Naive Bayes  is {:.2f}%".format(accuracy*100))



accuracy=accuracy_score(y_test,predictions2)

print("Accuracy of Logistic Regression  is {:.2f}%".format(accuracy*100))


accuracy=accuracy_score(y_test,predictions3 )

print("Accuracy of XGB Calssifieis {:.2f}%".format(accuracy*100))


import joblib

joblib.dump(model2,'final_pickle_model.pkl')

final_model = joblib.load('final_pickle_model.pkl')

pred = final_model.predict(X_test)
                          
accuracy=accuracy_score(y_test,pred)

print("Accuracy of Final Model is {:.2f}%".format(accuracy*100))


