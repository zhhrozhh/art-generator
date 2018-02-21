from keras.applications.vgg19 import VGG19
from tf_tools import *


v19 = VGG19(weights = 'imagenet',include_top = False)

class V19_CONV(MODEL):
    def __init__(self,name,W,H):
        MODEL.__init__(self,name)
        self.X = tf.placeholder(shape = [None,H,W,3],dtype = tf.float32)


        w0 = v19.get_layer('block1_conv1').get_weights()
        self.block1_conv1 = CONV2D(
            self.X,
            [3,3,3,64],
            name = 'block1_conv1',
            ini_val = w0[0],
            padding = 'SAME'
            )
        self.block1_conv1 = PLUSB(
            self.block1_conv1,
            name = 'block1_conv1',
            ini_val = w0[1]
            )
        self.block1_conv1 = tf.nn.relu(self.block1_conv1)

        w1 = v19.get_layer('block1_conv2').get_weights()
        self.block1_conv2 = CONV2D(
            self.block1_conv1,
            [3,3,64,64],
            name = 'block1_conv2',
            ini_val = w1[0],
            padding = 'SAME'
            )
        self.block1_conv2 = PLUSB(
            self.block1_conv2,
            name = 'block1_conv2',
            ini_val = w1[1]
            )
        self.block1_conv2 = tf.nn.relu(self.block1_conv2)

        self.block1_pool = tf.contrib.layers.max_pool2d(
            self.block1_conv2,
            kernel_size = (2,2),
            data_format = 'NHWC'
            )





        
        w2 = v19.get_layer('block2_conv1').get_weights()
        self.block2_conv1 = CONV2D(
            self.block1_pool,
            [3,3,64,128],
            name = 'block2_conv1',
            ini_val = w2[0],
            padding = 'SAME'
            )
        self.block2_conv1 = PLUSB(
            self.block2_conv1,
            name = 'block2_conv1',
            ini_val = w2[1]
            )
        self.block2_conv1 = tf.nn.relu(self.block2_conv1)

        w3 = v19.get_layer('block2_conv2').get_weights()
        self.block2_conv2 = CONV2D(
            self.block2_conv1,
            [3,3,128,128],
            name = 'block2_conv2',
            ini_val = w3[0],
            padding = 'SAME'
            )
        self.block2_conv2 = PLUSB(
            self.block2_conv2,
            name = 'block2_conv2',
            ini_val = w3[1]
            )
        self.block2_conv2 = tf.nn.relu(self.block2_conv2)

        self.block2_pool = tf.contrib.layers.max_pool2d(
            self.block2_conv2,
            kernel_size = (2,2),
            data_format = 'NHWC'
            )





        w2 = v19.get_layer('block3_conv1').get_weights()
        self.block3_conv1 = CONV2D(
            self.block2_pool,
            [3,3,128,256],
            name = 'block3_conv1',
            ini_val = w2[0],
            padding = 'SAME'
            )
        self.block3_conv1 = PLUSB(
            self.block3_conv1,
            name = 'block3_conv1',
            ini_val = w2[1]
            )
        self.block3_conv1 = tf.nn.relu(self.block3_conv1)

        w3 = v19.get_layer('block3_conv2').get_weights()
        self.block3_conv2 = CONV2D(
            self.block3_conv1,
            [3,3,256,256],
            name = 'block3_conv2',
            ini_val = w3[0],
            padding = 'SAME'
            )
        self.block3_conv2 = PLUSB(
            self.block3_conv2,
            name = 'block3_conv2',
            ini_val = w3[1]
            )
        self.block3_conv2 = tf.nn.relu(self.block3_conv2)

        w4 = v19.get_layer('block3_conv3').get_weights()
        self.block3_conv3 = CONV2D(
            self.block3_conv2,
            [3,3,256,256],
            name = 'block3_conv3',
            ini_val = w4[0],
            padding = 'SAME'
            )
        self.block3_conv3 = PLUSB(
            self.block3_conv3,
            name = 'block3_conv3',
            ini_val = w4[1]
            )
        self.block3_conv3 = tf.nn.relu(self.block3_conv2)

        w5 = v19.get_layer('block3_conv4').get_weights()
        self.block3_conv4 = CONV2D(
            self.block3_conv3,
            [3,3,256,256],
            name = 'block3_conv4',
            ini_val = w5[0]
            )
        self.block3_conv4 = PLUSB(
            self.block3_conv4,
            name = 'block3_conv4',
            ini_val = w5[1]
            )
        self.block3_conv4 = tf.nn.relu(self.block3_conv4)

        self.block3_pool = tf.contrib.layers.max_pool2d(
            self.block3_conv4,
            kernel_size = (2,2),
            data_format = 'NHWC'
            )





        w2 = v19.get_layer('block4_conv1').get_weights()
        self.block4_conv1 = CONV2D(
            self.block3_pool,
            [3,3,256,512],
            name = 'block4_conv1',
            ini_val = w2[0],
            padding = 'SAME'
            )
        self.block4_conv1 = PLUSB(
            self.block4_conv1,
            name = 'block4_conv1',
            ini_val = w2[1]
            )
        self.block4_conv1 = tf.nn.relu(self.block4_conv1)

        w3 = v19.get_layer('block4_conv2').get_weights()
        self.block4_conv2 = CONV2D(
            self.block4_conv1,
            [3,3,512,512],
            name = 'block4_conv2',
            ini_val = w3[0],
            padding = 'SAME'
            )
        self.block4_conv2 = PLUSB(
            self.block4_conv2,
            name = 'block4_conv2',
            ini_val = w3[1]
            )
        self.block4_conv2 = tf.nn.relu(self.block4_conv2)

        w4 = v19.get_layer('block4_conv3').get_weights()
        self.block4_conv3 = CONV2D(
            self.block4_conv2,
            [3,3,512,512],
            name = 'block4_conv3',
            ini_val = w4[0],
            padding = 'SAME'
            )
        self.block4_conv3 = PLUSB(
            self.block4_conv3,
            name = 'block4_conv3',
            ini_val = w4[1]
            )
        self.block4_conv3 = tf.nn.relu(self.block4_conv2)

        w5 = v19.get_layer('block4_conv4').get_weights()
        self.block4_conv4 = CONV2D(
            self.block4_conv3,
            [3,3,512,512],
            name = 'block4_conv4',
            ini_val = w5[0]
            )
        self.block4_conv4 = PLUSB(
            self.block4_conv4,
            name = 'block4_conv4',
            ini_val = w5[1]
            )
        self.block4_conv4 = tf.nn.relu(self.block4_conv4)

        self.block4_pool = tf.contrib.layers.max_pool2d(
            self.block4_conv4,
            kernel_size = (2,2),
            data_format = 'NHWC'
            )





        w2 = v19.get_layer('block5_conv1').get_weights()
        self.block4_conv1 = CONV2D(
            self.block4_pool,
            [3,3,512,512],
            name = 'block5_conv1',
            ini_val = w2[0],
            padding = 'SAME'
            )
        self.block5_conv1 = PLUSB(
            self.block4_conv1,
            name = 'block5_conv1',
            ini_val = w2[1]
            )
        self.block5_conv1 = tf.nn.relu(self.block5_conv1)

        w3 = v19.get_layer('block5_conv2').get_weights()
        self.block5_conv2 = CONV2D(
            self.block5_conv1,
            [3,3,512,512],
            name = 'block5_conv2',
            ini_val = w3[0],
            padding = 'SAME'
            )
        self.block5_conv2 = PLUSB(
            self.block5_conv2,
            name = 'block5_conv2',
            ini_val = w3[1]
            )
        self.block5_conv2 = tf.nn.relu(self.block5_conv2)

        w4 = v19.get_layer('block5_conv3').get_weights()
        self.block5_conv3 = CONV2D(
            self.block5_conv2,
            [3,3,512,512],
            name = 'block5_conv3',
            ini_val = w4[0],
            padding = 'SAME'
            )
        self.block5_conv3 = PLUSB(
            self.block5_conv3,
            name = 'block5_conv3',
            ini_val = w4[1]
            )
        self.block5_conv3 = tf.nn.relu(self.block5_conv2)

        w5 = v19.get_layer('block5_conv4').get_weights()
        self.block5_conv4 = CONV2D(
            self.block5_conv3,
            [3,3,512,512],
            name = 'block5_conv4',
            ini_val = w5[0]
            )
        self.block5_conv4 = PLUSB(
            self.block5_conv4,
            name = 'block5_conv4',
            ini_val = w5[1]
            )
        self.block5_conv4 = tf.nn.relu(self.block5_conv4)

        self.block5_pool = tf.contrib.layers.max_pool2d(
            self.block5_conv4,
            kernel_size = (2,2),
            data_format = 'NHWC'
            )



        self.train = None
        self.train_minib = None
        self.init = None
        
