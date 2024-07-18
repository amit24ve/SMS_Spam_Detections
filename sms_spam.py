import nltk
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

# msg=[line.strip() for line in open('SMSSpamCollection')]
# print(msg[0])
# for message in enumerate(msg[:10]):
#     print(message,end="\n")

msg=pd.read_csv("SMSSpamCollection",sep='\t',names=['label','message'])
# print(msg.head())
# print(msg.describe())
# print(msg.groupby('label').describe())
# msg['length']=msg['message'].apply(len)
# print(msg['length'])
# msg['length'].plot.hist(bins=50)
# msg.hist(column='length',by='label',bins=60,figsize=(12,6))
# plt.show()

mess="sample message! Notice: it has punctuation"
# no_punc=[c for c in mess if c not in string.punctuation]
# res=''.join(c for c in no_punc)
# print(res)

stopwords.words('english')
def test_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean words
    """
    nopunc=[c for c in mess if c not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# print(test_process.__doc__)
# print(test_process(mess))
msg['message'].head(5).apply(test_process)
# a=msg['message'].head(5).apply(test_process)
# print(a)
bow_transfermer=CountVectorizer(analyzer=test_process).fit(msg['message'])
# print(len(bow_transfermer.vocabulary_))

mess4=msg['message'][3]
# print(mess4)
bow4=bow_transfermer.transform([mess4])
# print(bow4.shape)
# message_bow=bow_transfermer.get_feature_names()[9554]
# print(message_bow)
mess_bow=bow_transfermer.transform(msg['message'])
# print(mess_bow.shape)
# print(mess_bow.nnz)

sparsity=(100.0*mess_bow.nnz/(mess_bow.shape[0]*mess_bow.shape[1]))
# print('sparsity:{}'.format(sparsity))

tfid_transformer=TfidfTransformer().fit(mess_bow)
tfid=tfid_transformer.transform(bow4)
# print(tfid)
# print(tfid_transformer.idf_[bow_transfermer.vocabulary_['university']])
msg_tfidf=tfid_transformer.transform(mess_bow)
# print(msg_tfidf)

s_d_m=MultinomialNB().fit(msg_tfidf,msg['label'])
# print(s_d_m.predict(tfid)[0])
all_pred=s_d_m.predict(msg_tfidf)
# print(all_pred)

msg_train,msg_test,label_train,label_test=train_test_split(msg['message'],msg['label'],test_size=0.3)
# print(msg_train)

pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=test_process)),
    ('tfid',TfidfTransformer()),
    # ('classifier',MultinomialNB())
    ('classifier',RandomForestClassifier())
])
pipeline.fit(msg_train,label_train)
predictions=pipeline.predict(msg_test)
# print(predictions)

# print(confusion_matrix(label_test,predictions))
print(classification_report(label_test,predictions))