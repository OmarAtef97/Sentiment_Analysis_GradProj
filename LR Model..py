#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import inflect
import re, string, unicodedata
import nltk 
import pickle
import tweepy
import matplotlib.pyplot as ply
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from nltk.corpus import stopwords
nltk.download('stopwords')


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
data_balanced


# In[4]:


data_balanced['sentiment'].value_counts()
sns.countplot('sentiment', data = data_balanced)


# In[5]:


print(nltk.corpus.stopwords.words('english'))


# In[6]:


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
        #last = re.sub('soooo', 'so', cleaner)
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


# In[51]:


x = data_balanced['new_text']
y = data_balanced['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
cv = CountVectorizer(binary=False, ngram_range=(1,2))
cv.fit(x_train)


# In[52]:


x_train_cv = cv.transform(x_train)
x_test_cv = cv.transform(x_test)
x_test_cv = x_test_cv.toarray()
x_train_cv = x_train_cv.toarray()


# In[53]:


k = len(cv.get_feature_names())


# In[54]:


model = LogisticRegression(solver='liblinear', C=0.05, multi_class=('ovr'),
                           random_state=0)
y_score = model.fit(x_train_cv, y_train)


# In[55]:


y_pred = model.predict(x_test_cv)
y_pred


# In[56]:


y_prob = model.predict_proba(x_test_cv)
y_prob


# In[57]:


#macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
#weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                     #average="weighted")
#macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  #average="macro")
#weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                     #average="weighted")
#def plot_roc_curve(fpr, tpr):
    #plt.plot(fpr, tpr, color='orange', label='ROC')
    #plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.legend()
    #plt.show()
#auc = roc_auc_score(y_test, y_prob)
#print('AUC: %.2f' % auc)


# In[58]:


model.score(x_train_cv, y_train)


# In[59]:


model.score(x_test_cv, y_test)


# In[60]:


import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs',  color='black')
ax.xaxis.set(ticks = (0, 1, 2))
ax.yaxis.set(ticks = (0, 1, 2))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[61]:


print(classification_report(y_test, y_pred))


# In[62]:


a = data_balanced['new_text']
b = data_balanced['sentiment']
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=0)


# In[63]:


tv = TfidfVectorizer(binary=False, ngram_range=(1,2))
tv.fit(a_train)


# In[64]:


a_train_tv = tv.transform(a_train)
a_test_tv = tv.transform(a_test)
a_test_tv = a_test_tv.toarray()
a_train_tv = a_train_tv.toarray()


# In[65]:


k1 =len(tv.get_feature_names())
k1


# In[66]:


model2 = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model2.fit(a_train_tv, b_train)


# In[67]:


y_pred2 = model2.predict(a_test_tv)
y_pred2


# In[68]:


model2.score(a_train_tv, b_train)


# In[69]:


model2.score(a_test_tv, b_test)


# In[70]:


import matplotlib.pyplot as plt

cm2 = confusion_matrix(b_test, y_pred2)
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm2)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs',  color='black')
ax.xaxis.set(ticks = (0, 1, 2))
ax.yaxis.set(ticks = (0, 1, 2))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm2[i, j], ha='center', va='center', color='white')
plt.show()


# In[71]:


print(classification_report(b_test, y_pred2))


# In[72]:


y_test = y_test.to_numpy()


# In[73]:


pos_count, neg_count, neu_count = 0, 0, 0
real_pos, real_neg, real_neu = 0, 0, 0

for i, prediction in enumerate(y_pred):
    if prediction == 1:
        pos_count += 1
    elif prediction == 0:
        neu_count += 1
    else:
        neg_count += 1
        
    if y_test[i] == 1:
        real_pos += 1
    elif y_test[i] == 0:
        real_neu += 1
    else:
        real_neg += 1

print('Actual positive:', real_pos)
print('Actual negative:', real_neg)
print('Actual neutral:', real_neu)
print('\n')
print('Positive predictions:', pos_count)
print('Negative predictons:', neg_count)
print('Neutral predictons:', neu_count)


# In[74]:


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


for i, prediction in enumerate(y_pred):
    if prediction == -1 and y_test[i] == -1:
        TP_pos_pos += 1
    elif prediction == 1 and y_test[i] == 1:
        TP_neu_neu += 1
    elif prediction == 0 and y_test[i] == 0:
        TP_neg_neg += 1
    elif prediction == -1 and y_test[i] == 1:
        FP_pos_neu += 1
    elif prediction == 2 and y_test[i] == 0:
        FP_pos_neg += 1
    elif prediction == 1 and y_test[i] == -1:
        FP_neu_pos += 1
    elif prediction == 1 and y_test[i] == 0:
        FP_neu_neg += 1
    elif prediction == 0 and y_test[i] == -1:
        FP_neg_pos += 1
    elif prediction == 0 and y_test[i] == 1:
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


# In[119]:


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


# In[120]:


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
        clean_word = re.sub('@[^ ]+', '', new)
        cleaner = re.sub('rt', '', clean_word)
        #last = re.sub('soooo', 'so', cleaner)
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

#def stem_words(data_twitter):
    #"""Stem words in list of tokenized words"""
    #stemmer = LancasterStemmer()
    #stems = []
    #for word in data_twitter:
        #stem = stemmer.stem(word)
        #stems.append(stem)
    #return stems

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
    #data_twitter = stem_words(data_twitter)
    data_twitter = lemmatize_verbs(data_twitter)
    return data_twitter

data_twitter['new_text'] = data_twitter['tweet'].apply(lambda x:' '.join(normalize(x.split())))
data_twitter =data_twitter.drop(['tweet'], axis=1)
data_twitter


# In[121]:


tv = TfidfVectorizer(binary=False, ngram_range=(1,3))
tv.fit(data_twitter['new_text'].values)


# In[122]:


train_data_twitter = tv.transform(data_twitter['new_text'])


# In[123]:


train_data_twitter = train_data_twitter.toarray()


# In[124]:


len(tv.get_feature_names())


# In[125]:


# model1
all_tweets = len(train_data_twitter)
shape = k
length = len(train_data_twitter[0])
app = shape - length
z = np.zeros((num_tweets, app), dtype=train_data_twitter.dtype)
train_data_twitter = np.c_[train_data_twitter, z]
print(all_tweets)
print(shape)
print(length)
print(app)


# In[127]:


predictions = model.predict(train_data_twitter)
predictions


# In[98]:


all_tweets = len(train_data_twitter)

pos, neg, neu = 0, 0, 0
pos_per, neg_per, neu_per = 0, 0, 0

for i, prediction in enumerate(predictions):
    if prediction == 2:
        pos += 1
    elif prediction == 1:
        neu += 1
    else:
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




