import csv
import sys
#sys.setdefaultendocing('utf-8')
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from nltk import pos_tag
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
#nltk.download()
smsdata = open('SMSSpamCollection','r')
csv_reader = csv.reader(smsdata, delimiter = '\t')
smsdata_data = []
smsdata_labels = []

for line in csv_reader:
    #print(line)
    smsdata_labels.append(line[0])
    smsdata_data.append(line[1])

smsdata.close()
# print the first 5 lines
for i in range(5):
    print(smsdata_data[i], smsdata_labels[i])

c= Counter(smsdata_labels)
print(c)

# 1. remove punctuations

def preprocessing(text):
    # check if standard punctuations, if so replace with blank or not replace
    text2 = ' '.join(''.join([' ' if ch in string.punctuation else ch for ch in text]).split())
    # tokenize the sentences into words based on white spaces and put the together as a list
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    # convert all cases to lower cases to reduce duplicate in corpus
    tokens = [word.lower() for word in tokens]
    # remove stop words
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    # stems the extra suffixes from the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # POS tagging, reduce word to the root
    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
    lemmatizer =  WordNetLemmatizer()

    def prat_lemmatize(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    pre_proc_text = ' '.join([prat_lemmatize(token, tag) for token, tag in tagged_corpus])
    return pre_proc_text


smsdata_data_2 = []
k =0
for i in smsdata_data:
    k+= 1
    smsdata_data_2.append(preprocessing(i))


trainset_size = int(round(len(smsdata_data_2)*0.7))

print('The training set size for this classifier is '+ str(trainset_size) + '\n')
x_train = np.array([''.join(rec) for rec in smsdata_data_2[0:trainset_size]])
y_train = np.array([rec for rec in smsdata_labels[0:trainset_size]])
x_test = np.array([''.join(rec) for rec in smsdata_data_2[trainset_size:len(smsdata_data_2)]])
y_test = np.array([rec for rec in smsdata_labels[trainset_size:len(smsdata_data_2)]])
#print(x_test)
#print(y_test)
# build TFIDF vectorizer
vectorizer = TfidfVectorizer(min_df = 2, ngram_range= (1,2), stop_words = 'english',
                             max_features= 4000, strip_accents= 'unicode')#, norm = '12'

x_train_2 = vectorizer.fit_transform(x_train).todense()
x_test_2 = vectorizer.transform(x_test).todense()

clf = MultinomialNB().fit(x_train_2,y_train)
ytrain_nb_predicted = clf.predict(x_train_2)
ytest_nb_predicted = clf.predict(x_test_2)

print('\nNaive Bayes -Train Confusion Matrix\n\n', pd.crosstab(y_train, ytrain_nb_predicted, rownames=
                                                               ['Acutuall'], colnames= ['Predicted']))
print('\nNaive Bayes -Train accuracy', round(accuracy_score(y_train, ytrain_nb_predicted),3))

print('\nNaive Bayes - Train classification Report\n', classification_report(y_train,ytrain_nb_predicted))

print('\nNaive Bayes -Test Confusion Matrix\n\n', pd.crosstab(y_test, ytest_nb_predicted, rownames=
                                                               ['Acutuall'], colnames= ['Predicted']))
print('\nNaive Bayes -Test accuracy', round(accuracy_score(y_test, ytest_nb_predicted),3))

print('\nNaive Bayes - Test classification Report\n', classification_report(y_test,ytest_nb_predicted))

# printing top features
feature_names = vectorizer.get_feature_names()
coefs = clf.coef_
intercept = clf.intercept_
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

print('\n\nTop 10 features -both first and last\n')
n = 10
top_n_coefs = zip(coefs_with_fns[:n], coefs_with_fns[:-(n+1):-1])
for (coefs_1, fn_1), (coefs_2, fn_2) in top_n_coefs:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coefs_1,fn_1, coefs_2, fn_2))
