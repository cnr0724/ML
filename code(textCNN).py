#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

data=pd.read_csv(r"C:\Users\Administrator\Desktop\data\data.csv")

applicant=data.iloc[0]
new_data=[]
for i in range(len(data)):
    if data.iloc[i,2]!="b''":
        new_data.append(data.iloc[i])
        
data=pd.DataFrame(new_data)

# In[2]:


import re

script=applicant[2]
escape_char=re.compile(r'\\x[0123456789abcdref]+')
script=re.sub(escape_char," ", script)
categories=[]
dic={}
data_length=len(data)

for i in range(data_length):
    if data.iloc[i,1] not in categories:
        num=len(categories)
        categories.append(data.iloc[i,1])
        dic[data.iloc[i,1]]=num
        
print(categories)
print(dic)


# In[3]:


label=data['Category']
x=data['Resume']


# In[4]:


x_handled=[]

for i in range(data_length):
    resume=x.iloc[i]
    resume=re.sub(escape_char, '', resume)
    resume=resume.replace('\\n',' ').replace('\n',' ')
    resume=resume[2:]
    x_handled.append(resume)


# In[5]:


df=pd.DataFrame(x_handled)


# In[6]:


df.to_csv(r"C:\Users\Administrator\Desktop\data\df.csv")


# In[7]:


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


# In[8]:


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


# In[9]:


df1=pd.DataFrame(x_handled)
df1.to_csv(r"C:\Users\Administrator\Desktop\data\df1.csv")


# In[10]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_handled)
vocab_size=len(tokenizer.word_index)+1

print(vocab_size)


# In[11]:


resumes_encoded=tokenizer.texts_to_sequences(x_handled)


# In[12]:


p_resumes=pad_sequences(resumes_encoded, maxlen=max([len(resume) for resume in resumes_encoded]), padding='post')


# In[13]:


max([len(resume) for resume in resumes_encoded])


# In[14]:


int_label=[]


# In[15]:


for i in range(len(label)):
    one_hot_vector=[0]*(len(dic))
    index=dic[label.iloc[i]]
    one_hot_vector[index]=1
    int_label.append(one_hot_vector)


# In[16]:


dataset=pd.DataFrame(list(zip(p_resumes, int_label)),columns=['x','label'])


# In[17]:


dataset.to_csv(r"C:\Users\Administrator\Desktop\data\p_data.csv")


# In[18]:


tok_dictionary=tokenizer.word_index

print(tok_dictionary)


# In[19]:


new_dictionary={}

regex3=re.compile('^[a-z][a-z][a-z]$',re.I)
regex4=re.compile('^[a-z][a-z]$', re.I)
for key in tok_dictionary:
    add1=True
    if regex1.match(key) or regex2.match(key) or regex3.match(key) or regex4.match(key):
        add1=False
        
    if add1==True:
        new_dictionary[key]=len(new_dictionary.keys())


# In[20]:


frequency_dictionary={}
for key in new_dictionary:
    frequency_dictionary[key]=0

    for i in range(len(x_handled)):
        re1=[]
        if key in x_handled[i].split():
            re1=re.findall('\\b'+key+'\\b',x_handled[i],flags=re.IGNORECASE)
        frequency_dictionary[key]+=len(re1)+1


# In[21]:


frequency_list=sorted(frequency_dictionary.items(), key=(lambda x:x[1]), reverse=True)


# In[22]:


pd.DataFrame(frequency_list).to_excel("C:\\Users\\Administrator\\Desktop\\data\\words.xls", encoding="utf-8")


# In[23]:


for i in range(len(frequency_list)):
    if frequency_list[i][1]<1000:
        print(i)
        break


# In[24]:


thousands=frequency_list[:6094]
new_dict={}
for tok in thousands:
    new_dict[tok[0]]=len(new_dict.keys())


# In[25]:


#tokenizer.word_index=new_dictionary
tokenizer.word_index=new_dict


# In[26]:


resumes=tokenizer.texts_to_sequences(x_handled)


# In[27]:


p_resume=pad_sequences(resumes, maxlen=max([len(resume) for resume in resumes]), padding='post')


# In[28]:


max([len(resume)for resume in resumes])


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import cnn_tool as tool


# In[50]:


x=p_resume
y=int_label
x_train, x_test, y_train, y_test = tool.divide(x,y,train_prop=0.8)


# In[51]:


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        """
        <Variable>
            - W: 각 단어의 임베디드 벡터의 성분을 랜덤하게 할당
        """
        #with tf.device('/gpu:0'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.initializers.GlorotUniform())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# In[52]:


flags=tf.compat.v1.flags
FLAGS = flags.FLAGS
flags_dict=FLAGS._flags()
keys_list=[keys for keys in flags_dict]
for keys in keys_list:
    FLAGS.__delattr__(keys)

flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedded vector (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

flags.DEFINE_string('f','','kernel')

import sys
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

import os
import time
import datetime

# 3. train the model and test
with tf.Graph().as_default():
    sess=tf.compat.v1.Session()
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=vocab_size,
                      embedding_size=FLAGS.embedding_dim,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.compat.v1.summary.histogram("{}".format(v.name), g)
                sparsity_summary = tf.compat.v1.summary.scalar("{}".format(v.name), tf.compat.v1.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.compat.v1.summary.scalar("loss", cnn.loss)
        acc_summary = tf.compat.v1.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]


        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        testpoint = 0
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.compat.v1.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                if testpoint + 100 < len(x_test):
                    testpoint += 100
                else:
                    testpoint = 0
                print("\nEvaluation:")
                dev_step(x_test[testpoint:testpoint+100], y_test[testpoint:testpoint+100], writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


# In[ ]:




