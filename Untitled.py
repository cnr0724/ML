#!/usr/bin/env python
# coding: utf-8

# Resume column 데이터 1차 전처리 및 라벨링

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# In[2]:


data=pd.read_csv(r"C:\Users\Administrator\Desktop\data\data.csv")


# In[3]:


data.head()


# In[4]:


applicant=data.iloc[0]
new_data=[]


# In[5]:


for i in range(len(data)):
    if data.iloc[i,2]!="b''":
        new_data.append(data.iloc[i])


# In[6]:


data=pd.DataFrame(new_data)


# In[7]:


import re


# In[8]:


script=applicant[2]
escape_char=re.compile(r'\\x[0123456789abcdref]+')
script=re.sub(escape_char," ", script)


# In[9]:


categories=[]
dic={}
data_length=len(data)


# In[10]:


for i in range(data_length):
    if data.iloc[i,1] not in categories:
        num=len(categories)
        categories.append(data.iloc[i,1])
        dic[data.iloc[i,1]]=num


# In[11]:


print(categories)
print(dic)


# In[12]:


label=data['Category']
x=data['Resume']


# In[13]:


x_handled=[]

for i in range(data_length):
    resume=x.iloc[i]
    resume=re.sub(escape_char, '', resume)
    resume=resume.replace('\\n',' ').replace('\n',' ')
    resume=resume[2:]
    x_handled.append(resume)


# In[14]:


df=pd.DataFrame(x_handled)


# In[15]:


df.to_csv(r"C:\Users\Administrator\Desktop\data\df.csv")


# 데이터 2차 전처리

# In[16]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
nltk.download('stopwords')
nltk.download('punkt')


# In[17]:


stop_words = stopwords.words('english')
regex1=re.compile('^[a-zA-Z]$')
regex2=re.compile('[0-9]+')
stop_words+=[',','(',')','|','$',';','%',':','.',('%c' % 39)]

for i in range(len(x_handled)):
    sentence_token=text_to_word_sequence(x_handled[i])
    result=[]
    for token in sentence_token:
        if token not in stop_words:
            if not regex2.match(token):
                if not regex1.match(token):
                    result.append(token)
    x_handled[i]=' '.join(result)


# In[18]:


df1=pd.DataFrame(x_handled)
df1.to_csv(r"C:\Users\Administrator\Desktop\data\df1.csv")


# Tokenizer 사용.

# In[19]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_handled)
vocab_size=len(tokenizer.word_index)+1

print(vocab_size)


# In[20]:


resumes_encoded=tokenizer.texts_to_sequences(x_handled)


# In[21]:


p_resumes=pad_sequences(resumes_encoded, maxlen=max([len(resume) for resume in resumes_encoded]), padding='post')


# In[22]:


max([len(resume) for resume in resumes_encoded])


# label 원핫벡터화

# In[23]:


int_label=[]


# In[24]:


for i in range(len(label)):
    one_hot_vector=[0]*(len(dic))
    index=dic[label.iloc[i]]
    one_hot_vector[index]=1
    int_label.append(one_hot_vector)


# 현재까지 상황 정리

# In[25]:


#x 데이터: p_resumes
#y 데이터: int_label


# In[26]:


dataset=pd.DataFrame(list(zip(p_resumes, int_label)),columns=['x','label'])


# In[27]:


dataset.to_csv(r"C:\Users\Administrator\Desktop\data\p_data.csv")


# In[28]:


#https://wikidocs.net/22933


# In[29]:


tok_dictionary=tokenizer.word_index

print(tok_dictionary)


# In[30]:


new_dictionary={}

regex3=re.compile('^[a-z][a-z][a-z]$',re.I)
regex4=re.compile('^[a-z][a-z]$', re.I)
for key in tok_dictionary:
    add1=True
    if regex1.match(key) or regex2.match(key) or regex3.match(key) or regex4.match(key):
        add1=False
        
    if add1==True:
        new_dictionary[key]=len(new_dictionary.keys())


# In[31]:


print(len(new_dictionary.keys()), len(tok_dictionary.keys()))


# In[32]:


frequency_dictionary={}
for key in new_dictionary:
    frequency_dictionary[key]=0

    for i in range(len(x_handled)):
        re1=[]
        if key in x_handled[i].split():
            re1=re.findall('\\b'+key+'\\b',x_handled[i],flags=re.IGNORECASE)
        frequency_dictionary[key]+=len(re1)+1


# In[33]:


frequency_dictionary


# In[34]:


frequency_list=sorted(frequency_dictionary.items(), key=(lambda x:x[1]), reverse=True)


# In[35]:


frequency_list 


# In[36]:


pd.DataFrame(frequency_list).to_excel("C:\\Users\\Administrator\\Desktop\\data\\words.xls", encoding="utf-8")


# In[37]:


for i in range(len(frequency_list)):
    if frequency_list[i][1]<1000:
        print(i)
        break


# In[38]:


thousands=frequency_list[:6094]
new_dict={}
for tok in thousands:
    new_dict[tok[0]]=len(new_dict.keys())


# In[39]:


#tokenizer.word_index=new_dictionary
tokenizer.word_index=new_dict


# In[40]:


len(tokenizer.word_index)


# In[41]:


resumes=tokenizer.texts_to_sequences(x_handled)


# In[42]:


p_resume=pad_sequences(resumes, maxlen=max([len(resume) for resume in resumes]), padding='post')


# In[43]:


max([len(resume)for resume in resumes])


# 모델 생성

# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# In[45]:


x=p_resume
y=int_label


# In[76]:


model=Sequential()
model.add(Embedding(6100,100))
model.add(LSTM(100))
model.add(Dense(25, activation='softmax'))


# In[77]:


es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('best_model.h5', monitor='val_acc',mode='max',verbose=1, save_best_only=True)


# In[78]:


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])


# In[80]:


x_arr=np.array(x)
y_arr=np.array(y)
x_train=x_arr[100:]
y_train=y_arr[100:]
x_test=x_arr[:100]
y_test=y_arr[:100]
history=model.fit(x_train,y_train,batch_size=128, epochs=3,
                  callbacks=[es,mc], validation_data=(x_train[:100],y_train[:100]))


# In[50]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfTransformer


# In[51]:


df_new=pd.DataFrame()
x_list=[]
for i in range(len(x_handled)):
    resum=[]
    words=text_to_word_sequence(x_handled[i])
    for word in words:
        if word in new_dict.keys():
            resum.append(word)
    
    x_list.append(' '.join(resum))

print(len(x))
print(len(y))
df_new['resume']=x_list
df_new['category_vec']=y


# In[52]:


num_label=[]
for i in range(len(int_label)):
    num_label.append(int_label[i].index(1))


# In[53]:


df_new['category_id']=num_label


# In[54]:


word_label=[]
inv_dic = {v: k for k, v in dic.items()}
for i in range(len(num_label)):
    word_label.append(inv_dic[num_label[i]])


# In[55]:


df_new['category_word']=word_label


# In[56]:


import matplotlib.pyplot as plt


# In[82]:


fig=plt.figure(figsize=(8,6))
colors=['grey','grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','orange','orange','orange','orange','orange']
df_new.groupby('category_word').resume.count().sort_values().plot.barh(ylim=0,color=colors,title='Distribution')
plt.xlabel('Number of occurrences', fontsize=10)


# In[58]:


tfidf=TfidfVectorizer(sublinear_tf=True,min_df=5,ngram_range=(1,2),stop_words='english')

features=tfidf.fit_transform(df_new.resume).toarray()

labels=df_new.category_id


# In[59]:


print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


# In[83]:


N = 5
for category_word, category_id in sorted(dic.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("\n==> %s:" %(category_word))
    print("  * Most Correlated Unigrams: %s" %(', '.join(unigrams[-N:])))
    print("  * Most Correlated Bigrams: %s" %(', '.join(bigrams[-N:])))

features_chi2=chi2(features, labels == 0)
indices=np.argsort(features_chi2[0])
feature_names=np.array(tfidf.get_feature_names())[indices]


# In[112]:


X=df_new['resume']
y_list=[]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=0)


# In[113]:


models=[ RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[114]:


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc


# In[115]:


X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df_new.index, test_size=0.3, 
                                                               random_state=1)

model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[116]:


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("LinearSVC Accuracy: %.2f%%" % (accuracy * 100.0))


# In[124]:


from sklearn import metrics


print('\t\t   CLASSIFICATION METRICS FOR SVM\n')
print(metrics.classification_report(y_test, y_pred, 
                                    target_names= df_new['category_word'].unique()))
print("LinearSVC Accuracy: %.2f%%" % (accuracy * 100.0))


# In[118]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dic.keys(), 
            yticklabels=dic.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16);


# In[119]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[120]:


model1=XGBClassifier()
model1.fit(X_train,y_train)


# In[121]:


y_pred=model1.predict(X_test)


# In[128]:


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

print('\t\t   CLASSIFICATION METRICS FOR XGB\n')
print(metrics.classification_report(y_test, y_pred, 
                                    target_names= df_new['category_word'].unique()))
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))


# In[123]:


conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dic.keys(), 
            yticklabels=dic.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - XGBoost\n", size=16);


# In[127]:


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))


# linearSVC가 낫구나....

# In[74]:


import pickle

filename='finalModel.sav'
pickle.dump(model, open(filename,'wb'))

