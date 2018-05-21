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
#####CHECK THE REGULAR TRAIN FOR MOST DETAILED COMMENTS. HERE I JUST CHANGE THE COMENTS FROM THE TRAINING WHILE LOOP


SEQLEN = 270
BATCHSIZE = 1000
ALPHASIZE = 6
BETASIZE = 3756
INTERNALSIZE = 100
NLAYERS = 1
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
L=200
char_class=3756

with open ('chlist', 'rb') as fp:
    chlist = pickle.load(fp)


with open ('lbl1h', 'rb') as fp:
    lbl1h = pickle.load(fp)


usls=[0,0,0,0,0,0]
first=True
i=0
for i in range (len(chlist)):
    lstc=chlist[i]
    while len(lstc)<270:
        lstc.append(usls)
print("\n done loading stuff \n")

with tf.device('/device:GPU:0'):
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')
    Xf = tf.placeholder(tf.float32, [None, None, 6], name='Xf')    # [ BATCHSIZE, SEQLEN ]
    Y_ = tf.placeholder(tf.int32, [None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, BETASIZE, 1.0, 0.0)
    Hinf = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hinf')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    Hinb = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hinb')
    cellfwd = rnn.GRUCell(INTERNALSIZE)
    dcellfwd = rnn.DropoutWrapper(cellfwd,input_keep_prob=0.8)
    cellbwd = rnn.GRUCell(INTERNALSIZE)
    dcellbwd = rnn.DropoutWrapper(cellbwd,input_keep_prob=0.8)
    Yr, H = tf.nn.bidirectional_dynamic_rnn(dcellfwd,dcellbwd, Xf, dtype=tf.float32, initial_state_fw=Hinf,initial_state_bw=Hinb)
    H = tf.identity(H, name='H')  # just to give it a name
    Yr = tf.identity(Yr, name='Yr')
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




chlist,lbl1h=unison_shuffled_list(chlist,lbl1h)
for i in range(50000+1):
    if i>10000:
        learning_rate=0.0005
    if i>20000:
    	learning_rate=0.0003
    xf=np.asarray(chlist[ind:ind2])
    #xb=xf[:,::-1,:]
    sp_index=[]
    y=np.asarray(lbl1h[ind:ind2])
    for it in range(len(y)):
        if y[it]<=1200:
            sp_index.append(it)
    for j in range(len(y)):
        if y[j]>1200:
            y[j]=randint(0,3755)    #stochastic cmcl. Change any out of distribution classes to a random value after a while.
    ind=ind2
    ind2=ind2+BATCHSIZE
    if ind2>len(chlist):
        ind2=BATCHSIZE
        ind=0
        epoch+=1
        chlist,lbl1h=unison_shuffled_list(chlist,lbl1h)
        print( "EPOCH ADVANCED, NOW IS : " + str(epoch) )
    feed_dict = {Xf: xf, Y_: y, Hinf: istate, Hinb: istate2, lr: learning_rate, batchsize: BATCHSIZE}
    if init_all==False:
        _, yu, ostate,cellf,cellb = sess.run([train_step, Yf, H,cellfwd.variables,cellbwd.variables], feed_dict=feed_dict)
		l2=sess.run(sess.graph.get_tensor_by_name('fully_connected_1/weights:0'))
		b2=sess.run(sess.graph.get_tensor_by_name('fully_connected_1/biases:0' ))
		l1=sess.run(sess.graph.get_tensor_by_name('fully_connected/weights:0'))
		b1=sess.run(sess.graph.get_tensor_by_name('fully_connected/biases:0' ))
        step=step+1
        init_all=True
    else:
        _, yu, ostate = sess.run([train_step, Yf, H], feed_dict=feed_dict)
    if i % 20 == 0:
        a, c,pred = sess.run([accuracy, cross_entropy,correct_prediction],{Xf: xf, Y_: y, Hinf: istate, Hinb: istate2})
        real_pred=[]
        for it in range(len(sp_index)):
            real_pred.append(pred[sp_index[it]])    
        sp_ac=np.mean(real_pred)
        print(str(i) + ": total accuracy:" + str(a) + " speciliazed accuracy: "+ str(sp_ac) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
    istate=ostate[0]
    istate2=ostate[1]

l2a=sess.run(sess.graph.get_tensor_by_name('fully_connected_1/weights:0'))
b2a=sess.run(sess.graph.get_tensor_by_name('fully_connected_1/biases:0' ))
l1a=sess.run(sess.graph.get_tensor_by_name('fully_connected/weights:0'))
b1a=sess.run(sess.graph.get_tensor_by_name('fully_connected/biases:0' ))
cellfa= sess.run(cellfwd.variables)
cellba=sess.run(cellbwd.variables)
with open('cellsandlayersCMCL.pkl', 'wb') as f:
    pickle.dump([cellf, cellb, l1,l2,b1,b2, cellfa,cellba,l1a,l2a,b1a,b2a,istate,istate2], f)

