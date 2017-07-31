import tensorflow as tf
import globals_static as gs
import get_data.PrepareMnistData as gp

class MnistFlow:
    def __init__(self, layerNum, inputChannel, outputChannels, filterSize):
        self.layerCnt = layerNum
        if (self.layerCnt > len(outputChannels)):
            print ("Invalid input: channels are not enough")
            exit(1)
        self.filterSize = filterSize
        self.inChan = inputChannel
        self.outChan = outputChannels[:]
        self.weights = []
        self.biases = []
        self.fc_weights = []
        self.fc_biases = []
    
    def __buildConvLayer(self, layer, inChan):
        self.weights.append(tf.Variable(tr.truncated_normal([self.filterSize, self.filterSize, inChan, self.outChan[layer]],
                                            stddev = 0.1, seed = SEED)))
        self.biases.append(tf.Variable(tf.zeros[self.outChan[layer]]))

    def __buildFcLayer(self, layer, uSize, depth):
        self.fc_weights.append(tf.Variable(tf.truncated_normal([uSize, depth],
                                            stddev = 0.1, seed = SEED)))
        self.fc_biases.append(tf.Variable(tf.constant(0.1, shape[depth])))

    def buildModel(self, validate_data, test_data):
        self.train_data_node = tf.placeholder(tf.float32, shape=(gs.BATCH_SZIE, gs.IMAGE_SIZE, gs.IMAGE_SIZE, gs.NUM_CHANNELS))
        self.train_label_node = tf.placeholder(tf.float32, shape=(gs.BATCH_SZIE, gs.NUM_CHANNELS))
        self.validate_data_node = tf.constant(validate_data)
        self.test_data_node = tf.constant(test_data)
        inChan = gs.NUM_CHANNELS
        for i in range(self.layerCnt):
            self.__buildConvLayer(i, inChan)
            inChan = self.outChan[i]
        self.__buildFcLayer(0, gs.IMAGE_SIZE//4*gs.IMAGE_SIZE//4*64, 512)
        self.__buildFcLayer(1, 512, gs.NUM_LABELS)
        print("build model done")
    
    def model(self, data, isTrain=False):
        inData = data
        for i in range(self.layerCnt):
            conv = tf.nn.conv2d(inData, self.weights[i], 
                                strides=[1,1,1,1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases[i]))
            pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            inData = pool
        pool_shape = inData.get_shape().as_list()
        reshape = tf.reshape(inDta, pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3])
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc_weights[0]) + self.fc_biases[0])
            
        if isTrain:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, self.fc_weights[1]) + self.fc_biases[1]
        print("train model done")



