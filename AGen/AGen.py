from tf_tools import V19_CONV
import gc
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


class Style_Transfer(V19_CONV):
    def __init__(self,name,W,H,style_path,content_path):
        V19_CONV.__init__(self,name,W,H)

        self.W = W
        self.H = H
        self.style = np.asarray(Image.open(style_path).resize((W,H)))/256
        self.style.resize(1,H,W,3)
        
        self.content = np.asarray(Image.open(content_path).resize((W,H)))/256
        self.content.resize(1,H,W,3)
        
        self.init = tf.global_variables_initializer()
        
            
        self.AC1 = tf.placeholder(shape = [1,None,None,512],dtype = tf.float32)# 4 2
        self.AC2 = tf.placeholder(shape = [1,None,None,256],dtype = tf.float32)# 3 2
        self.AC3 = tf.placeholder(shape = [1,None,None,128],dtype = tf.float32)# 2 2
        
        self.AS1 = tf.placeholder(shape = [1,None,None,256],dtype = tf.float32)# 3 3
        self.AS2 = tf.placeholder(shape = [1,None,None,512],dtype = tf.float32)# 4 3
        self.AS3 = tf.placeholder(shape = [1,None,None,128],dtype = tf.float32)# 2 1
        
        
        self.c_cost1 = tf.reduce_mean(tf.squared_difference(self.AC1,self.block4_conv2))/4
        self.c_cost2 = tf.reduce_mean(tf.squared_difference(self.AC2,self.block3_conv2))/4
        self.c_cost3 = tf.reduce_mean(tf.squared_difference(self.AC3,self.block2_conv2))/4
        
        self.c_cost = 0.33*self.c_cost1 + 0.33*self.c_cost2 + 0.33*self.c_cost3
        
        u1 = tf.reshape(self.AS1,[-1,256])
        U1 = tf.reshape(self.block3_conv3,[-1,256])
        Gu1 = tf.matmul(tf.transpose(u1),u1)
        GU1 = tf.matmul(tf.transpose(U1),U1)
        self.s_cost1 = tf.reduce_sum(tf.squared_difference(Gu1,GU1))/(4*(W*H*3)**2)
        
        u2 = tf.reshape(self.AS2,[-1,512])
        U2 = tf.reshape(self.block4_conv3,[-1,512])
        Gu2 = tf.matmul(tf.transpose(u2),u2)
        GU2 = tf.matmul(tf.transpose(U2),U2)
        self.s_cost2 = tf.reduce_sum(tf.squared_difference(Gu2,GU2))/(4*(W*H*3)**2)
        
        
        u3 = tf.reshape(self.AS3,[-1,128])
        U3 = tf.reshape(self.block2_conv1,[-1,128])
        Gu3 = tf.matmul(tf.transpose(u3),u3)
        GU3 = tf.matmul(tf.transpose(U3),U3)
        self.s_cost3 = tf.reduce_sum(tf.squared_difference(Gu3,GU3))/(4*(W*H*3)**2)
        
        self.s_cost = 0.33*self.s_cost1+0.33*self.s_cost2+0.33*self.s_cost3
        
        self.cost = 40*self.s_cost + 16*self.c_cost
    def getSC(self):
                 
        with tf.Session() as sess:
            sess.run(self.init)
            ASS = [self.block3_conv3,self.block4_conv3,self.block2_conv1]
            
            AS1,AS2,AS3 = sess.run(ASS,feed_dict = {self.X:self.style})
            
            ACS = [self.block4_conv2,self.block3_conv2,self.block2_conv2]
            AC1,AC2,AC3 = sess.run(ACS,feed_dict = {self.X:self.content})
            
        return AS1,AS2,AS3,AC1,AC2,AC3



def denoiser(X,method = 'gaussian'):
    #ss = X.shape
    X = X.reshape(X.shape[1:])
    assert method in ['gaussian','nlmean','none']
    s = X.std()
    if method == 'gaussian':
        return scipy.ndimage.filters.gaussian_filter(X,sigma = s).reshape(1,*X.shape)
    if method == 'nlmean':
        return nlmeans(X,sigma = s/10,block_radius = 1).reshape(1,*X.shape)
    return X.reshape(1,*X.shape)



def VGG19Gen(style,content,size = None,iter = 600,denoise = 'gaussian'):
    assert denoise in ['gaussian','nlmean','none']
    if size is None:
        W,H = Image.open(content).size
    else:
        W,H = size
    gen = Style_Transfer('test',W,H,style,content)
    AS1,AS2,AS3,AC1,AC2,AC3 = gen.getSC()
    sx = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = gen.content
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.show()
        fig.canvas.draw()
        dic = {
            gen.X:x,
            gen.AS1:AS1,
            gen.AS2:AS2,
            gen.AS3:AS3,
            gen.AC1:AC1,
            gen.AC2:AC2,
            gen.AC3:AC3
        }
        grd = tf.gradients(gen.cost,[gen.X])
        lr = 0.524
        m = np.zeros(x.shape)
        v = np.zeros(x.shape)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lri = lr
        cost_a = 10000
        for i in range(1,iter+1):
            gc.collect()
            dic[gen.X] = x
            cost,dX = sess.run([gen.cost,grd[0]],feed_dict = dic)
            if cost > cost_a:
                lr = lr*0.995
            cost_a = cost
            lri = lr*np.sqrt(1-beta2**i)/(1-beta1**i)
            m = beta1*m + (1-beta1)*dX
            v = beta2*v + (1-beta2)*dX*dX
            me = m/(1-beta1**i)
            ve = v/(1-beta2**i)
            x = x - lri*me/(np.sqrt(ve)+eps)

            x = denoiser(x,method = denoise)
            sx = np.copy(x)            
            ax.clear()
            sx[sx>1] = 1
            sx[sx<0] = 0
            #print(sx.shape)
            ax.imshow(sx[0,:,:,:])
            ax.set_title(str(i)+'/'+str(iter)+' cost:'+str(cost))
            fig.canvas.draw()
    return sx[0]
