import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def ACT(inp,name = "defACT",get_trainable = False):
    bshape = inp.shape[1:]
    R = tf.get_variable(name+"R",shape=bshape,initializer=tf.contrib.layers.xavier_initializer())
    LR = tf.get_variable(name+"LR",shape=bshape,initializer=tf.contrib.layers.xavier_initializer())
    r = tf.nn.sigmoid(R)
    RES = (tf.nn.tanh(inp)*r+tf.nn.relu(inp)*(1-r))*LR
    if get_trainable:
        tlist = {name+"R":R,name+"LR":LR}
        return tlist,RES
    return RES

def ACT_S(inp,name = "defACTS",get_trainable = False):
    bshape = inp.shape[1:]
    R = tf.get_variable(name+"R",shape=bshape,initializer=tf.contrib.layers.xavier_initializer())
    r = tf.nn.sigmoid(R)
    RES = tf.nn.tanh(inp)*r+tf.nn.relu(inp)*(1-r)
    if get_trainable:
        tlist = {name+"R":R}
        return tlist,RES
    return RES

def BN(
    inp,
    name = "defBN",
    decay = 0.99999,
    axes = None,
    epsilon = 10e-5,
    get_trainable = False
    ):
    nD = len(inp.shape)
    if axes == None:
        axes = list(range(nD-1))
    BNmean,BNvar = tf.nn.moments(inp,axes = axes)
    scale = tf.get_variable(name+"scale",shape = inp.shape[1:],initializer = tf.contrib.layers.xavier_initializer())
    shift = tf.get_variable(name+"shift",shape = inp.shape[1:],initializer = tf.contrib.layers.xavier_initializer())
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    def BNupdate():
        ema_apply_op = ema.apply([BNmean,BNvar])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(BNmean),tf.identity(BNvar)
    BNmean,BNvar = BNupdate()
    RES = tf.nn.batch_normalization(inp,BNmean,BNvar,shift,scale,epsilon)
    if get_trainable:
        tlist = {name+"scale":scale,name+"shift":shift}
        return tlist,RES
    return RES


def CONV_RES(
    inp,
    name="defRES",
    step=1,
    f=3,
    bn=False,
    get_trainable = False,
    act = False
    ):
    b,w,h,c = inp.shape
    X = inp
    tlist = {}
    for i in range(step):
        if bn:
            X = BN(X,name = name+"BNX"+str(i))
        W = tf.get_variable(name+"W"+str(i),shape = [f,f,c,c],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(name+"b"+str(i),shape = [w,h,c],initializer = tf.zeros_initializer())
        Z = tf.nn.conv2d(X,W,strides = [1,1,1,1],padding = "SAME")
        if act:
            X = ACT_S(Z)
        else:
            X = tf.nn.relu(Z)
    if bn:
        X = BN(X,name = name+"BNN")
    W = tf.get_variable(name+"W",shape = [f,f,c,c],initializer = tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(name+"b",shape = [w,h,c],initializer = tf.zeros_initializer())
    RES = tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="SAME")+inp+b)
    return RES

def GOOG(
    inp,
    name = "defGOOG",
    channels = 1,
    get_trainable = False
    ):
    b,w,h,c = inp.shape
    w110 = tf.get_variable(name+"w110",shape = [1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A110 = tf.nn.conv2d(inp,w110,strides = [1,1,1,1],padding = "SAME")
    
    w113 = tf.get_variable(name+"w113",shape = [1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    Z113 = tf.nn.conv2d(inp,w113,strides = [1,1,1,1],padding = "SAME")
    w33 = tf.get_variable(name+"w33",shape = [3,3,channels,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A33 = tf.nn.conv2d(Z113,w33,strides = [1,1,1,1],padding = "SAME")
    
    w115 = tf.get_variable(name+"w115",shape = [1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    Z115 = tf.nn.conv2d(inp,w115,strides = [1,1,1,1],padding = "SAME")
    w55 = tf.get_variable(name+"w55",shape = [5,5,channels,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A55 = tf.nn.conv2d(Z115,w55,strides = [1,1,1,1],padding = "SAME")
    
    mx = tf.nn.max_pool(inp,ksize = [1,3,3,1],strides = [1,1,1,1],padding="SAME")
    wm = tf.get_variable(name + "wm",shape = [1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    Am = tf.nn.conv2d(mx,wm,strides = [1,1,1,1],padding = "SAME")
    
    RES = tf.concat([A110,A33,A55,Am],axis = 3)
    return RES


def YSW(
    inp,
    name = "defYSW",
    channels = 1,
    get_trainable = False
    ):
    b,w,h,c = inp.shape
    w11 = tf.get_variable(name + "w11",shape=[1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A11 = tf.nn.conv2d(inp,w11,strides = [1,1,1,1],padding = "SAME")
    
    w33 = tf.get_variable(name + "w33",shape = [3,3,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A33 = tf.nn.conv2d(inp,w33,strides = [1,1,1,1],padding = "SAME")
    
    w55 = tf.get_variable(name + "w55",shape = [5,5,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    A55 = tf.nn.conv2d(inp,w55,strides = [1,1,1,1],padding = "SAME")
    
    Amx0 = tf.nn.max_pool(inp,ksize=[1,3,3,1],strides=[1,1,1,1],padding="SAME")
    Wmx = tf.get_variable(name + "WMX",shape = [1,1,c,channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    Amx = tf.nn.conv2d(Amx0,Wmx,strides = [1,1,1,1],padding = "SAME")
    
    w0 = tf.get_variable(name+"w0",shape=[1],initializer=tf.ones_initializer())
    w1 = tf.get_variable(name+"w1",shape=[1],initializer=tf.ones_initializer())
    w2 = tf.get_variable(name+"w2",shape=[1],initializer=tf.ones_initializer())
    w3 = tf.get_variable(name+"w3",shape=[1],initializer=tf.ones_initializer())
    
    RES = A11*w0 + A33*w1 + A55*w2 + Amx*w3

    return RES

def CONV2D(
    inp,
    wshape,
    name = "defCONV2D",
    padding = "VALID",
    strides = [1,1,1,1],
    get_trainable = False,
    ini_val = None
    ):
    if ini_val is not None:
        W = tf.get_variable(name+"W",initializer=tf.constant(ini_val))
    else:
        W = tf.get_variable(name+"W",shape=wshape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    RES = tf.nn.conv2d(inp,W,strides=strides,padding = padding)
    return RES

def PLUSB(
    inp,
    name = "defPLUSB",
    get_trainable = False,
    ini_val = None
    ):
    if ini_val is not None:
        b = tf.get_variable(name+"b",initializer=tf.constant(ini_val))
    elif len(inp.shape) == 4:
        _,w,h,c = inp.shape
        b = tf.get_variable(name+"b",shape = [w,h,c],initializer = tf.zeros_initializer())
    elif len(inp.shape) == 2:
        _,c = inp.shape
        b = tf.get_variable(name+"b",shape = [1,c],initializer = tf.zeros_initializer())
    return inp + b

def FC(
    inp,
    outdim,
    name = "defFC",
    get_variable = False,
    ini_val = None
    ):
    _,c = inp.shape
    if ini_val is not None:
        W = tf.get_variable(name + 'W',initializer = tf.constant(ini_val))
    else:
        W = tf.get_variable(name+"W",shape = [c,outdim],initializer = tf.contrib.layers.xavier_initializer())
    RES = tf.matmul(inp,W)
    return RES

def MMlize(X,Y):
    YY0 = Y
    XX0 = X
    XXMAX = XX0[YY0[:,0].argmax():YY0[:,0].argmax()+1,:,:,:]
    YYMAX = YY0[YY0[:,0].argmax():YY0[:,0].argmax()+1,:]
    XXMIN = XX0[YY0[:,1].argmin():YY0[:,1].argmin()+1,:,:,:]
    YYMIN = YY0[YY0[:,1].argmin():YY0[:,1].argmin()+1,:]
    XX0 = np.concatenate([XXMAX,XX0,XXMIN])
    YY0 = np.concatenate([YYMAX,YY0,YYMIN])
    return XX0,YY0
    
class MODEL:
    def __init__(self,name="def"):
        self.name = name
        self.X = None
        self.Y = None
        self.OUP = None
        self.init = tf.global_variables_initializer()
        self.cost = None
        self.opt = None
    def open(self,init=True):
        self.sess=tf.Session()
        if init:
            self.sess.run(self.init)
    def close(self,reset=True):
        self.sess.close()
        tf.reset_default_graph()
    def train(self,X,Y,loop=300,mmmode = False):
        loss = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()

        if mmmode:
            X,Y = MMlize(X,Y)

        for i in range(loop):
            _,cost = self.sess.run([self.opt,self.cost],feed_dict={self.X:X,self.Y:Y})
            loss.append(cost)
            if len(loss)>50:
                loss = loss[1:]
            ax.clear()
            ax.plot(loss)
            ax.set_title(str(i)+"/"+str(loop))
            fig.canvas.draw()
    def train_minib(self,X,Y,dis=False,loop=300,bloop=4,bsize=128,mmmode = False):
        loss = []
        L = len(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()
        for i in range(loop):
            idx = np.random.choice(range(L),size = bsize)
            X_ = X[idx,:,:,:]
            Y_ = Y[idx,:]
            if mmmode:
                X_,Y_ = MMlize(X_,Y_)
            for b in range(bloop):
                _,cost = self.sess.run([self.opt,self.cost],feed_dict={self.X:X_,self.Y:Y_})
                loss.append(cost)
                if len(loss) > 50:
                    loss = loss[1:]
                ax.clear()
                ax.plot(loss)
                ax.set_title(str(i)+"/"+str(loop))
                fig.canvas.draw()
    def save(self,name = None):
        if name == None:
            name = self.name
        sv = tf.train.Saver()
        sv.save(self.sess,name)
    def load(self,name = None):
        if name == None:
            name = self.name
        sv = tf.train.Saver()
        sv.restore(self.sess,name)

