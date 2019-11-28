from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_viterbi_accuracy
import pickle
import random
import pandas as pd
import numpy as np
import tokenization
from tokenization import *
from keras import backend as K
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


EMBEDDING_DIMENSION=1024
#BertLoaded

words=set()
with open("NCBI_data_train.txt.pkl","rb") as F:
    data=pickle.load(F)
    #data=data[:10]
# with open("kp20k_validation.pkl","rb") as F:
#     data_val=pickle.load(F)

with open("NCBI_data_train.txt.pkl","rb") as F:
    data_test=pickle.load(F)

# def balance(data):
#     for j in range(len(data)):
#         X=data[j][0]
#         y=data[j][1]
#         cnt=min([len([i for i in range(len(y)) if y[i]=="O"]),
#                 sum([i=="I-Disease" for i in y]),sum([])])

for i in range(len(data)):
    abstract=data[i][0]
    for word in abstract:
        words.add(word)

#words = list(set(data["Word"].values))
words.add("ENDPAD")
words=list(words)

n_words = len(words)

tags = ["O","I-Disease","B-Disease"]
n_tags = len(tags) 

max_len = 25
word2idx = {w: i + 1 for i, w in enumerate(words)}
tokenizer = FullTokenizer('BioELMO/vocabulary.txt')
for i,w in enumerate(words):
    try:
        word2idx[word]=tokenizer.convert_tokens_to_ids(word)[0]
    except:
        word2idx[word]=tokenizer.convert_tokens_to_ids(word)
tag2idx = {t: i for i, t in enumerate(tags)}

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "BioELMO/biomed_elmo_options.json"
weight_file = "BioELMO/biomed_elmo_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
print("ok0")
elmo = Elmo(options_file, weight_file, 1)
print("ok")

def init_embedding():

    embedding_matrix = np.zeros((n_words+1,EMBEDDING_DIMENSION))
    count=0
    print("ok")
    for w in words:
        try:
            
            embeddings = elmo(batch_to_ids([[w]]))
            #import pdb;pdb.set_trace()
            embedding_vector = embeddings["elmo_representations"][0][0]
            embedding_vector=embedding_vector[0]
            #embedding_vector = pretrained_word_embeddings(torch.LongTensor([word2idx[w]]))#modelBert(torch.LongTensor([[word2idx[w]]]))[1][0]
            #embedding_vector.tolist()
            #embedding_vector=embedding_vector[0]
            #print(embedding_vector)
            embedding_matrix[word2idx[w]] = embedding_vector.detach().numpy()
        except Exception as e:
            print(e)
            count+=1
            embedding_matrix[word2idx[w]] = [random.random() for j in range(2)]

    return embedding_matrix
print("ok1")
embedding_matrix=init_embedding()


#print(count,len(t.word_index.items()))

from keras.preprocessing.sequence import pad_sequences
# print(data[0])
# exit(0)
X = [[word2idx[w] for w in s[0]] for s in data]
#print(X)
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=25)

X_test = []
for s in data_test:
    temp=[]
    for w in s[0]:
        try:
            temp.append(word2idx[w])
        except:
            temp.append(0)
    X_test.append(temp)
X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=25)


y = [[tag2idx[w] for w in s[1]] for s in data]
y_test = [[tag2idx[w] for w in s[1]] for s in data_test]

y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=0)
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)

from keras.utils import to_categorical
y = np.array([to_categorical(i, num_classes=n_tags) for i in y])
y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])
# print(y[0].shape)
# exit(0)
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=EMBEDDING_DIMENSION,
                  weights=[embedding_matrix],input_length=max_len, mask_zero=True
                  ,trainable=False)(input)  # 2-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0))(model)  # variational biLSTM
model = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # a dense layer as suggested by neuralNer
#crf = CRF(n_tags)  # CRF layer
#out = crf(model)  # output

model = Model(input, model)
from keras import optimizers
#sgd = optimizers.SGD(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True)


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc",recall_m,precision_m,f1_m])#[crf.accuracy])
print(model.summary())

model.fit(X, np.array(y), epochs=1,
                    validation_split=0.1)

# from keras import backend as K


from sklearn.metrics import precision_score,recall_score,f1_score

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out
    
# evaluate the model
#test_pred=model.predict(X_test)

# from sklearn.metrics import classification_report

# y_pred = model.predict(X_test, batch_size=64, verbose=1)
# y_pred_bool = np.argmax(y_pred, axis=1)
# print(y_pred)
#print(classification_report(y_test, y_pred_bool))
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("loss:%f accuracy:%f f1_score:%f precision:%f recall:%f"%(loss, accuracy, f1_score, precision, recall))
#pred_labels = pred2label(test_pred)
#test_labels = pred2label(y_test)
#print(pred_labels)
#print(classification_report(test_labels,pred_labels))
# f1,pr,rec=0,0,0
# for i in range(len(test_labels)):
#     pr+=precision_score(test_labels[i],pred_labels[i],average='micro')
#     f1+=f1_score(test_labels[i],pred_labels[i],average='micro')
#     rec+=recall_score(test_labels[i],pred_labels[i],average='micro')
# pr/=len(test_labels)
# rec/=len(test_labels)
# f1/=len(test_labels)
#print("Precision %f, Recall %f, F1 %f,"%(pr,rec,f1))
# print("loss %f, accuracy %f, f1_score %f, precision %f, recall %f"%(loss, accuracy, f1_score, precision, recall))
# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
# test_pred = model.predict(X_test, verbose=1)


# print(test_labels[0],pred_labels[0])
# print("F1-score: {:.1%}".format(f1_score(test_labels[0], pred_labels[0])))
# print(classification_report(test_labels, pred_labels))


