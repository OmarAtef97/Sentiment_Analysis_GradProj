#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import inflect
import keras
import pydot as pyd
import numpy as np
import seaborn as sns
import tensorflow
import tensorflow as tf
import re, string, unicodedata
import nltk 
import matplotlib.pyplot as ply
import matplotlib.pyplot as plt
import tweepy
import pickle
from nltk import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.svm import SVC
from tensorflow.keras import layers
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Embedding, Dropout, SimpleRNN
from sklearn.metrics import mean_squared_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
nltk.download('stopwords')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cols = ['sentiment','id','date','query_string','user','tweet']
df = pd.read_csv('F:\\Gam3a\\Courses\\Deep Learning\\Tweets-1.6M.csv', encoding='latin1', names=cols)
df1 = pd.read_csv("F:\\Gam3a\\GP\\Dataset\\English\\new_english_test.csv", encoding='latin1')
df2 = pd.read_csv("F:\\Gam3a\\GP\\Dataset\\English\\english_submission.csv", encoding='latin1')
df1['sentiment'] = df2['sentiment']
df1_new = df1.drop(['id'], axis=1)
np.random.seed(0)
index = np.random.randint(low=0, high=1599999, size=10000)
data1 = df.loc[index, ['sentiment', 'tweet']].reset_index(drop=True)
data1['sentiment'] = data1['sentiment'].replace({4:1})
data1['sentiment'] = data1['sentiment'].replace({0:-1})
frames = [df1_new, data1]
data = pd.concat(frames, sort=True)
data['sentiment'].value_counts()


# In[3]:


data_pos = data[data['sentiment'] == 1 ]
data_neg = data[data['sentiment'] == -1]
data_neu = data[data['sentiment'] == 0 ]
positive_data = data_pos.head(len(data_pos) - 5133)
negative_data = data_neg.head(len(data_neg) - 3738)
neutral_data = data_neu
data_balanced = pd.concat([positive_data, negative_data, neutral_data])
data_balanced = shuffle(data_balanced)

data_balanced = data_balanced.reset_index(drop=True)
data_balanced['sentiment'] = data_balanced['sentiment'].replace({1:'positive'})
data_balanced['sentiment'] = data_balanced['sentiment'].replace({0:'neutral'})
data_balanced['sentiment'] = data_balanced['sentiment'].replace({-1:'negative'})
print(data_balanced)

print(sns.countplot('sentiment', data = data_balanced))


# In[4]:


print(nltk.corpus.stopwords.words('english'))


# In[5]:


pat1 = '@[^ ]+'
pat2 = 'http[^ ]+'
pat3 = 'www.[^ ]+'
pat4 = '#[^ ]+'

combined_pat = '|'.join((pat1, pat2, pat3, pat4))

clean_tweets_balanced = []

def remove_non_ascii(tweet):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in tweet:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(tweet):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in tweet:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(tweet):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in tweet:
        new_word = re.sub(r'[^\w\s]', '', word)
        newest_word = re.sub('btw', 'by the way', new_word)
        new = re.sub('_', '', new_word)
        clean_word = re.sub('@[^ ]+', '', new)
        cleaner = re.sub('RT', '', clean_word)
        if cleaner != '':
            new_words.append(cleaner)
    return new_words

def replace_numbers(tweet):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in tweet:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(tweet):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in tweet:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

#def stem_words(tweet):
    #"""Stem words in list of tokenized words"""
    #stemmer = LancasterStemmer()
    #stems = []
    #for word in tweet:
        #stem = stemmer.stem(word)
        #stems.append(stem)
    #return stems

def lemmatize_verbs(tweet):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in tweet:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(tweet):
    tweet = remove_non_ascii(tweet)
    tweet = to_lowercase(tweet)
    tweet = remove_punctuation(tweet)
    tweet = replace_numbers(tweet)
    tweet = remove_stopwords(tweet)
    #tweet = stem_words(tweet)
    tweet = lemmatize_verbs(tweet)
    return tweet

data_balanced['new_text'] = data_balanced['tweet'].apply(lambda x:' '.join(normalize(x.split())))
data_balanced = data_balanced.drop(['tweet'], axis=1)
data_balanced


# In[6]:


tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(data_balanced['new_text'].values)
X = tokenizer.texts_to_sequences(data_balanced['new_text'].values)
X = pad_sequences(X)
X[:5]


# In[7]:


x = data_balanced['new_text']
y = data_balanced['sentiment']
y = pd.get_dummies(data_balanced['sentiment']).values
y


# In[8]:


print(data_balanced['sentiment'][0])
print(data_balanced['sentiment'][1])
print(data_balanced['sentiment'][2])
print('\n')
print(y[0])
print(y[1])
print(y[2])
print('\n')
print(np.argmax(y[0]))
print(np.argmax(y[1]))
print(np.argmax(y[2]))


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[58]:


X_train


# In[59]:


max_len=0
for i in X:
    if(max_len<len(i)):
        max_len=len(i)
print("The maximum number of words in a single document is : " , max_len)


# In[60]:


y_train


# In[61]:


from keras.optimizers import SGD
model_1 = Sequential()
model_1.add(Embedding(5000, 256, input_length = X.shape[1]))
model_1.add(Dropout(0.2))
model_1.add(GRU(units=256, return_sequences=True, dropout= 0.4, activation='relu'))
model_1.add(SimpleRNN(units=256, activation = 'relu'))
model_1.add(Dense(3, activation='softmax'))
#loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
#opt = SGD(lr=0.01, momentum=0.9)
model_1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()


# In[62]:


plot_model(model_1, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)


# In[63]:


history = model_1.fit(X_train, y_train, epochs=10, batch_size=64, verbose = 2)


# In[64]:


loss, accuracy = model_1.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))


# In[65]:


loss, accuracy = model_1.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[66]:


#plt.style.use('ggplot')

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])

#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train','test'], loc='upper left')
#plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])

#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train','test'], loc='upper left')
#plt.show()


# In[67]:


predictions = model_1.predict(X_test)
predictions


# In[68]:


modulated_pred = predictions.round().astype(int)
modulated_pred


# In[69]:


pos_count, neg_count, neu_count = 0, 0, 0
real_pos, real_neg, real_neu = 0, 0, 0

for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2:
        pos_count += 1
    elif np.argmax(prediction)==1:
        neu_count += 1
    else:
        neg_count += 1
        
    if np.argmax(y_test[i])==2:
        real_pos += 1
    elif np.argmax(y_test[i])==1:
        real_neu += 1
    else:
        real_neg += 1

print('Positive predictions:', pos_count)
print('Negative predictons:', neg_count)
print('Neutral predictons:', neu_count)
print('\n')
print('Actual positive:', real_pos)
print('Actual negative:', real_neg)
print('Actual neutral:', real_neu)


# In[70]:


TP_pos, TP_neg, TP_neu, TN_pos, TN_neg, TN_neu = 0, 0, 0, 0, 0, 0
FP_pos, FN_pos, FP_neg, FN_neg, FP_neu, FN_neu = 0, 0, 0, 0, 0, 0
TP_pos_pos, TP_neu_neu, TP_neg_neg = 0, 0, 0
FP_pos_neu, FP_pos_neg = 0, 0
FP_neg_neu, FP_neg_pos = 0, 0
FP_neu_pos, FP_neu_neg = 0, 0

TP, FP, TN, FN = 0, 0, 0, 0

false_pos = abs(real_pos - pos_count)
false_neg = abs(real_neg - neg_count)
false_neu = abs(real_neu - neu_count)

true_pos = min(real_pos, pos_count)
true_neg = min(real_neg, neg_count)
true_neu = min(real_neu, neu_count)

acc = (true_pos+true_neg+true_neu)/(true_pos+true_neg+true_neu+false_pos+false_neg+false_neu)


for i, prediction in enumerate(predictions):
    if np.argmax(prediction)==2 and np.argmax(y_test[i])==2:
        TP_pos_pos += 1
    elif np.argmax(prediction)==1 and np.argmax(y_test[i])==1:
        TP_neu_neu += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==0:
        TP_neg_neg += 1
    elif np.argmax(prediction)==2 and np.argmax(y_test[i])==1:
        FP_pos_neu += 1
    elif np.argmax(prediction)==2 and np.argmax(y_test[i])==0:
        FP_pos_neg += 1
    elif np.argmax(prediction)==1 and np.argmax(y_test[i])==2:
        FP_neu_pos += 1
    elif np.argmax(prediction)==1 and np.argmax(y_test[i])==0:
        FP_neu_neg += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==2:
        FP_neg_pos += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==1:
        FP_neg_neu += 1
        
TP_pos = TP_pos_pos
TN_pos = TP_neg_neg + FP_neg_neu + FP_neu_neg + TP_neu_neu
FP_pos = FP_pos_neg + FP_pos_neu
FN_pos = FP_neg_pos + FP_neu_pos

TP_neg = TP_neg_neg
TN_neg = TP_pos_pos + FP_pos_neu + FP_neu_pos + TP_neu_neu
FP_neg = FP_neg_pos + FP_neg_neu
FN_neg = FP_pos_neg + FP_neu_neg

TP_neu = TP_neu_neu
TN_neu = TP_pos_pos + FP_pos_neg + FP_neg_pos + TP_neg_neg
FP_neu = FP_neu_pos + FP_neu_neg
FN_neu = FP_pos_neu + FP_neg_neu

TP = TP_pos + TP_neg + TP_neu
TN = TN_pos + TN_neg + TN_neu
FP = FP_pos + FP_neg + FP_neu
FN = FN_pos + FN_neg + FN_neu

precision_pos = TP_pos / (TP_pos + FP_pos)
precision_neg = TP_neg / (TP_neg + FP_neg)
precision_neu = TP_neu / (TP_neu + FP_neu)
precision = (precision_pos + precision_neg + precision_neu) / 3

recall_pos = TP_pos / (TP_pos + FN_pos)
recall_neg = TP_neg / (TP_neg + FN_neg)
recall_neu = TP_neu / (TP_neu + FN_neu)
recall = (recall_pos + recall_neg + recall_neu) / 3

f1_score_pos = 2 * ((precision_pos * recall_pos)/(precision_pos + recall_pos))
f1_score_neg = 2 * ((precision_neg * recall_neg)/(precision_neg + recall_neg))
f1_score_neu = 2 * ((precision_neu * recall_neu)/(precision_neu + recall_neu))
f1_score = (f1_score_pos + f1_score_neg + f1_score_neu) / 3

accuracy = (TP + TN) / (TP + TN + FP + FN)

print('positive precision:', precision_pos)
print('negative precision:', precision_neg)
print('neutral precision:', precision_neu)
print('Overall Precision:', precision)
print('\n')
print('positive recall:', recall_pos)
print('negative recall:', recall_neg)
print('neutral recall:', recall_neu)
print('Overall Recall:', recall)
print('\n')
print('positive f1_score:', f1_score_pos)
print('negative f1_score:', f1_score_neg)
print('neutral f1_score:', f1_score_neu)
print('Overall F1_score:', f1_score)
print('\n')
print('Accuracy:', accuracy)
print('Overall Sentiment Classification Accuracy:', acc)


# In[71]:


len(predictions)


# In[72]:


len(X_test)


# In[25]:


model_2 = Sequential()
model_2.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(3, activation='softmax'))
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.summary()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[27]:


model_2.fit(X_train, y_train, epochs=170, batch_size=64, verbose = 2, validation_data=(X_test,y_test))


# In[28]:


loss, accuracy = model_2.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))


# In[29]:


loss, accuracy = model_2.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[30]:


y_pred2 = model_2.predict(X_test)
y_pred2


# In[31]:


modulated_pred2 = y_pred2.round().astype(int)
modulated_pred2


# In[32]:


len(y_pred2)


# In[33]:


pos_count, neg_count, neu_count = 0, 0, 0
real_pos, real_neg, real_neu = 0, 0, 0
for i, prediction in enumerate(y_pred2):
    if np.argmax(prediction)==1:
        pos_count += 1
    elif np.argmax(prediction)==0:
        neu_count += 1
    else:
        neg_count += 1
        
    if np.argmax(y_test[i])==1:
        real_pos += 1
    elif np.argmax(y_test[i])==0:
        real_neu += 1
    else:
        real_neg += 1
        
print('Positive predictions: ', pos_count)
print('Negative predictions: ', neg_count)
print('Neutral predictions: ', neu_count)

print('Real positive: ', real_pos)
print('Real negative: ', real_neg)
print('Real neutral: ', real_neu)


# In[56]:


len(y_test)


# In[73]:


from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['red', 'blue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i+1, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
print('Class 1: Negative')
print('Class 2: Neutral')
print('Class 3: Positive')


# In[74]:


TP_pos, TP_neg, TP_neu, TN_pos, TN_neg, TN_neu = 0, 0, 0, 0, 0, 0
FP_pos, FN_pos, FP_neg, FN_neg, FP_neu, FN_neu = 0, 0, 0, 0, 0, 0
TP_pos_pos, TP_neu_neu, TP_neg_neg = 0, 0, 0
FP_pos_neu, FP_pos_neg = 0, 0
FP_neg_neu, FP_neg_pos = 0, 0
FP_neu_pos, FP_neu_neg = 0, 0

TP, FP, TN, FN = 0, 0, 0, 0


for i, prediction in enumerate(y_pred2):
    if np.argmax(prediction)==1 and np.argmax(y_test[i])==1:
        TP_pos_pos += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==0:
        TP_neu_neu += 1
    elif np.argmax(prediction)==2 and np.argmax(y_test[i])==2:
        TP_neg_neg += 1
    elif np.argmax(prediction)==1 and np.argmax(y_test[i])==0:
        FP_pos_neu += 1
    elif np.argmax(prediction)==1 and np.argmax(y_test[i])==2:
        FP_pos_neu += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==1:
        FP_neu_pos += 1
    elif np.argmax(prediction)==0 and np.argmax(y_test[i])==2:
        FP_neu_neg += 1
    elif np.argmax(prediction)==2 and np.argmax(y_test[i])==1:
        FP_neg_pos += 1
    elif np.argmax(prediction)==2 and np.argmax(y_test[i])==0:
        FP_neg_neu += 1
        
TP_pos = TP_pos_pos
TN_pos = TP_neg_neg + FP_neg_neu + FP_neu_neg + TP_neu_neu
FP_pos = FP_pos_neg + FP_pos_neu
FN_pos = FP_neg_pos + FP_neu_pos

TP_neg = TP_neg_neg
TN_neg = TP_pos_pos + FP_pos_neu + FP_neu_pos + TP_neu_neu
FP_neg = FP_neg_pos + FP_neg_neu
FN_neg = FP_pos_neg + FP_neu_neg

TP_neu = TP_neu_neu
TN_neu = TP_pos_pos + FP_pos_neg + FP_neg_pos + TP_neg_neg
FP_neu = FP_neu_pos + FP_neu_neg
FN_neu = FP_pos_neu + FP_neg_neu

TP = TP_pos + TP_neg + TP_neu
TN = TN_pos + TN_neg + TN_neu
FP = FP_pos + FP_neg + FP_neu
FN = FN_pos + FN_neg + FN_neu

precision_pos = TP_pos / (TP_pos + FP_pos)
precision_neg = TP_neg / (TP_neg + FP_neg)
precision_neu = TP_neu / (TP_neu + FP_neu)
precision = (precision_pos + precision_neg + precision_neu) / 3

recall_pos = TP_pos / (TP_pos + FN_pos)
recall_neg = TP_neg / (TP_neg + FN_neg)
recall_neu = TP_neu / (TP_neu + FN_neu)
recall = (recall_pos + recall_neg + recall_neu) / 3

f1_score_pos = 2 * ((precision_pos * recall_pos)/(precision_pos + recall_pos))
f1_score_neg = 2 * ((precision_neg * recall_neg)/(precision_neg + recall_neg))
f1_score_neu = 2 * ((precision_neu * recall_neu)/(precision_neu + recall_neu))
f1_score = (f1_score_pos + f1_score_neg + f1_score_neu) / 3

accuracy = (TP + TN) / (TP + TN + FP + FN)

print('positive precision:', precision_pos)
print('negative precision:', precision_neg)
print('neutral precision:', precision_neu)
print('overall precision:', precision)
print('\n')
print('positive recall:', recall_pos)
print('negative recall:', recall_neg)
print('neutral recall:', recall_neu)
print('overall recall:', recall)
print('\n')
print('positive f1_score:', f1_score_pos)
print('negative f1_score:', f1_score_neg)
print('neutral f1_score:', f1_score_neu)
print('overall f1_score:', f1_score)
print('\n')
print('accuracy', accuracy)


# In[75]:


model_1.save('Sentiment Analysis using RNN & GRU')


# In[76]:


consumer_key = 'Ny1M6IdrXliLOPJp0CLg6jwY8'
consumer_secret = 'tnzL7FB8I2bZC2LHGge2piCC0MbF2feHY54fCjKBPDfJCOl1AV'
access_token = '350057086-rY5dmqLXkWivwKlOxf7dHmsrocBm1HMVq1rNsmKP'
access_token_secret = 'aZrGdUmgovUpG54V8jis1PA3mZ0VjdpurTNgKf0cTTqhB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

a=input("Enter Tag: ")
tweets = api.search(a, count=200)
a=[]
for tweet in tweets:
    if tweet.lang == "en":
        a.append(tweet.text)
        
data_twitter = pd.DataFrame(a, columns=['tweet'])
num_tweets = len(data_twitter)
data_twitter


# In[77]:


pat1 = '@[^ ]+'
pat2 = 'http[^ ]+'
pat3 = 'www.[^ ]+'
pat4 = '#[^ ]+'

combined_pat = '|'.join((pat1, pat2, pat3, pat4))

clean_tweets_twitter = []

for t in data_twitter['tweet']:
    t = t.lower()
    stripped = re.sub(combined_pat, ' ', t)
    negations = re.sub("n't", "not", stripped)

    clean_tweets_twitter.append(negations)

data_twitter['tweet'] = clean_tweets_twitter

def remove_non_ascii(data_twitter):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in data_twitter:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(data_twitter):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in data_twitter:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(data_twitter):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in data_twitter:
        new_word = re.sub(r'[^\w\s]', '', word)
        newest_word = re.sub('btw', 'by the way', new_word)
        new = re.sub('_', '', new_word)
        clean_word = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', new)
        cleaner = re.sub('rt', '', clean_word)
        if cleaner != '':
            new_words.append(cleaner)
    return new_words

def replace_numbers(data_twitter):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in data_twitter:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(data_twitter):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in data_twitter:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(data_twitter):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in data_twitter:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(data_twitter):
    data_twitter = remove_non_ascii(data_twitter)
    data_twitter = to_lowercase(data_twitter)
    data_twitter = remove_punctuation(data_twitter)
    data_twitter = replace_numbers(data_twitter)
    data_twitter = remove_stopwords(data_twitter)
    data_twitter = lemmatize_verbs(data_twitter)
    return data_twitter

data_twitter['text'] = data_twitter['tweet'].apply(lambda x:' '.join(normalize(x.split())))
data_twitter =data_twitter.drop(['tweet'], axis=1)
data_twitter


# In[78]:


tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(data_twitter['text'].values)
X_twitter = tokenizer.texts_to_sequences(data_twitter['text'].values)
X_twitter = pad_sequences(X_twitter)
X_twitter[:5]


# In[79]:


length = len(X_twitter[0])
length


# In[80]:


all_tweets = len(X_twitter)
shape = len(X_test[0])
length = len(X_twitter[0])
app = shape - length
z = np.zeros((num_tweets, app), dtype=X_twitter.dtype)
X_twitter = np.c_[X_twitter, z]


# In[81]:


predictions_twitter = model_2.predict(X_twitter)
modulated_pred3 = predictions_twitter.round().astype(int)
modulated_pred3


# In[82]:


pos, neg, neu = 0, 0, 0
pos_per, neg_per, neu_per = 0, 0, 0

for i, prediction in enumerate(predictions_twitter):
    if np.argmax(prediction)==2:
        pos += 1
    elif np.argmax(prediction)==1:
        neu += 1
    elif np.argmax(prediction) ==0:
        neg += 1
        
pos_per = (pos/all_tweets)*100
neg_per = (neg/all_tweets)*100
neu_per = (neu/all_tweets)*100
        
print('Positive predictions:', pos)
print('Negative predictons:', neg)
print('Neutral predictons:', neu)
print('\n')
print('Positive percentage:', pos_per, '%')
print('Negative percentage:', neg_per, '%')
print('Neutral percentage:', neu_per, '%')
print('\n')

if pos > neg and pos > neu:
    print('Overall Sentiment is Positive.')
elif neg > pos and neg > neu:
    print('Overall Sentiment is Negative.')
else:
    print('Overall Sentiment is Neutral.')


# In[ ]:





# In[ ]:





# In[ ]:




