import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import math
import numpy as np
from random import randint
import random

tf.set_random_seed(0)

SEQLEN = 270
BATCHSIZE = 1000
ALPHASIZE = 6
BETASIZE = 3756
INTERNALSIZE = 100
NLAYERS = 1
learning_rate = 0.0001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
L=200
char_class=3756

with open ('chlist_t', 'rb') as fp:
    chlist_t = pickle.load(fp)


with open ('lbl1h_t', 'rb') as fp:
    lbl1h_t = pickle.load(fp)

with open('cellsandlayersCMCL.pkl','rb') as f:  # Python 3: open(..., 'rb')
    cellf, cellb,l1,l2,b1,b2, cellfa,cellba,l1a,l2a,b1a,b2a,istate,istate2 = pickle.load(f)

usls=[0,0,0,0,0,0]
first=True
i=0
for i in range (len(chlist_t)):
    lstc_t=chlist_t[i]
    while len(lstc_t)<270:
        lstc_t.append(usls)


with tf.device('/device:GPU:0'):
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')
    Xf = tf.placeholder(tf.float32, [None, None, 6], name='Xf')    # [ BATCHSIZE, SEQLEN ]
    Y_ = tf.placeholder(tf.int32, [None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, BETASIZE, 1.0, 0.0)
    #yrtt,htst = sess.run([Yr, H], feed_dict=feed_dict)
    # input state
    Hinf = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hinf')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    Hinb = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hinb')
    # How to properly apply dropout in RNNs: see README.md
    cellfwd = rnn.GRUCell(INTERNALSIZE)
    dcellfwd = rnn.DropoutWrapper(cellfwd,input_keep_prob=1)
    cellbwd = rnn.GRUCell(INTERNALSIZE)
    dcellbwd = rnn.DropoutWrapper(cellbwd,input_keep_prob=1)
    #multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    # "naive dropout" implementation
    #dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
    #multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    #multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer
    Yr, H = tf.nn.bidirectional_dynamic_rnn(dcellfwd,dcellbwd, Xf, dtype=tf.float32, initial_state_fw=Hinf,initial_state_bw=Hinb)
    # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
    # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
    H = tf.identity(H, name='H')  # just to give it a name
    Yr = tf.identity(Yr, name='Yr')
    #Hf = tf.identity(H[0], name='Hf')  # just to give it a name
    #Hb = tf.identity(H[1], name='Hb')
    Yrta2=tf.add(Yr[0],Yr[1],name='Yrta2')
    Yrta2=tf.div(Yrta2,2)
    Ypoo=tf.layers.average_pooling1d(Yrta2,[3],strides=[3])
    Yrr = tf.reshape(Ypoo, [BATCHSIZE,INTERNALSIZE*90],name='Yrr')
    Y1=tf.contrib.layers.fully_connected(Yrr,L,activation_fn=tf.nn.sigmoid)
    Y2=tf.contrib.layers.fully_connected(Y1,char_class,activation_fn=None)
    Yf = tf.nn.softmax(Y2)
    entropy_list = -tf.log(BETASIZE+0.)-tf.reduce_mean(tf.log(Yf),1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y2, labels=Yo_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    correct_prediction = tf.equal(tf.argmax(Yf, 1), tf.argmax(Yo_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    cp_grad=tf.train.AdamOptimizer(lr).compute_gradients(cross_entropy)



istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
istate2 = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
step = 0
epoch=0

init_all=False
ind=0
ind2=BATCHSIZE
start_time=time.time()


def unison_shuffled_list(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a,b



ind=0
ind2=BATCHSIZE
st1=0
ac=0
e_l1=0
out_e1=0
sp_ac1=0
sp_ent1=0
sp_ent=0
ent1=0
fp=0
fp1=0
fn=0
fn1=0


sess.run(tf.assign(sess.graph.get_tensor_by_name('fully_connected_1/weights:0'),l2a))
sess.run(tf.assign(sess.graph.get_tensor_by_name('fully_connected/weights:0'),l1a))
sess.run(tf.assign(sess.graph.get_tensor_by_name('fully_connected_1/biases:0'),b2a))
sess.run(tf.assign(sess.graph.get_tensor_by_name('fully_connected/biases:0'),b1a))
sess.run(tf.assign(cellfwd.variables[0],cellfa[0]))
sess.run(tf.assign(cellfwd.variables[1],cellfa[1]))
sess.run(tf.assign(cellfwd.variables[2],cellfa[2]))
sess.run(tf.assign(cellfwd.variables[3],cellfa[3]))
sess.run(tf.assign(cellbwd.variables[0],cellba[0]))
sess.run(tf.assign(cellbwd.variables[1],cellba[1]))
sess.run(tf.assign(cellbwd.variables[2],cellba[2]))
sess.run(tf.assign(cellbwd.variables[3],cellba[3]))


tsh=1.5

ror_test=[]
ror_i=[]
for it in range(len(chlist_t)):
    if lbl1h_t[it]<=1200:
        ror_test.append(chlist_t[it])
        ror_i.append(lbl1h_t[it])
cnt=0
it=0
while cnt<17173:
    if lbl1h_t[it]>1200:
        ror_test.append(chlist_t[it])
        ror_i.append(lbl1h_t[it])
        cnt=cnt+1
    it=it+1

ind=0
ind2=BATCHSIZE
st1=0
ac=0
e_l1=0
out_e1=0
sp_ac1=0
sp_ent1=0
sp_ent=0
ent1=0
fp=0
fp1=0
fn=0
fn1=0
ror1=0
ror_test,ror_i=unison_shuffled_list(ror_test,ror_i)

while(True):
    xf=np.asarray(ror_test[ind:ind2])
    #xb=xf[:,::-1,:]
    y=np.asarray(ror_i[ind:ind2])
    #xb=xf[:,::-1,:]
    sp_index=[]
    y=np.asarray(ror_i[ind:ind2])
    for it in range(len(y)):
        if y[it]<=1200:
            sp_index.append(it)
    ind=ind2
    ind2=ind2+BATCHSIZE
    if ind2>len(ror_i):
        break
    feed_dict = {Xf: xf, Y_: y, Hinf: istate, Hinb: istate2, lr: learning_rate, batchsize: BATCHSIZE}
    a1, c,pred,e_l = sess.run([accuracy, cross_entropy,correct_prediction,entropy_list],{Xf: xf, Y_: y, Hinf: istate, Hinb: istate2})
    real_pred=[]
    real_ent=[]
    ##########
    for it in range(len(sp_index)):
        if e_l[sp_index[it]]<tsh:
            real_ent.append(e_l[sp_index[it]])
            cnt=cnt+1
            continue    
        real_pred.append(pred[sp_index[it]])    
        real_ent.append(e_l[sp_index[it]])
    ##########
    for it in range(len(pred)):
        if sp_index.count(it):
            cnt=cnt+1
            continue
        if e_l[it]<tsh:
            pred[it]=True
            fn=fn+1
            continue
        real_pred.append(False)
        fp=fp+1
    ##########
    a1=np.mean(pred)
    ror=len(real_pred)/len(pred)
    den=fp+fn
    fp=fp/(den)
    fn=fn/(den)
    sp_ac=np.mean(real_pred)
    sp_ent=np.mean(real_ent)
    out_e=(sum(e_l)-sum(real_ent))/(len(e_l)-len(real_ent))
    e_l=np.mean(e_l)
    print(str(i) + ": total accuracy:" + str(a1) + " speciliazed accuracy: "+ str(sp_ac) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
    ac=ac+a1
    sp_ac1=sp_ac1+sp_ac
    sp_ent1=sp_ent1+sp_ent
    e_l1=e_l1+e_l
    out_e1=out_e1+out_e
    ror1=ror1+ror
    fn1=fn1+fn
    fp1=fp1+fp
    st1=st1+1


print(" \n **final accuracy** = "+str(sp_ac1/(st1-1)) )