from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

"""
    check if the data (zip) file is already downloaded
    if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz
"""
if not isfile('cifar-10-python.tar.gz'):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


# In[2]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


# In[3]:


def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

    plt.imshow(sample_image)


# In[54]:


#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np

# Explore the dataset
batch_id = 3
sample_id = 9999
features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
imageTest = features[0:200]

display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


# In[5]:


features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
sample_image = features[0]
#plt.imshow(sample_image)
X_test = features[0:200]
y_test = labels[0:200]
X_test.shape


# In[6]:


from sklearn.utils import shuffle

X_test, y_test = shuffle(X_test, y_test)


# In[7]:


import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128


# In[13]:

import os
session = tf.Session()

sigma = 0.1
conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = 0, stddev = sigma))
fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = sigma))
fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = 0 , stddev = sigma))

session.run(tf.global_variables_initializer())

def restoreWeights():
    for wIdx in range(0, len(weightsConvMat)):
        assign_op = weightsConvMat[wIdx].assign(wOrigConv[wIdx])
        session.run(assign_op)

    for wIdx in range(0, len(weightsFcMat)):
        assign_op = weightsFcMat[wIdx].assign(wOrigFc[wIdx])
        session.run(assign_op)



wConvOrig1 = session.run(conv1_w)
wConvOrig2 = session.run(conv2_w)
wFcOrig1 = session.run(fc1_w)
wFcOrig2 = session.run(fc2_w)
wFcOrig3 = session.run(fc3_w)


saver = tf.train.Saver()
save_dir = 'checkpoints_lenetMinst/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')
#session.run(tf.global_variables_initializer())
init = tf.global_variables_initializer()
session.run(init)
try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

wOrigConv = [wConvOrig1, wConvOrig2 ]
wOrigFc = [wFcOrig1, wFcOrig2, wFcOrig3]


from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }


    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.

    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    #conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    #fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    #fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    #fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits, conv1, conv2, fc1, fc2


# In[15]:


a = tf.placeholder(tf.float32, (200,32,32,3))
x_image = tf.reshape(a, [-1, 32, 32, 1])

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
init = tf.global_variables_initializer()
session.run(init)


# In[16]:

rate = 0.001

logits, conv1, conv2, fc1, fc2= LeNet(x_image)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

### check point;; this is where we start execution ###

MEFullyConnected1Graphs = []
MEFullyConnected2Graphs = []
MEFullyConnected3Graphs = []
MEConv1Graphs = []
MEConv2Graphs = []

weightsConvMat = [conv1_w, conv2_w]
weightsFcMat = [fc1_w, fc2_w, fc3_w]


wConv1 = session.run(conv1_w)
wConv2 = session.run(conv2_w)
wFc1 = session.run(fc1_w)
wFc2 = session.run(fc2_w)
wFc3 = session.run(fc3_w)

wConv = [wConv1, wConv2]
wFc = [wFc1, wFc2, wFc3]
session.run(tf.global_variables_initializer())

def printTensorArray(wtListConv):
    ## this is assuming this is of the type wConv
    num = len(wtListConv)

    for wtMat in wtListConv:
        shape = wtMat.shape
        xlen = shape[0]
        ylen = shape[1]
        numChnls = shape[2]
        numFilters = shape[3]

        for c in range(0,numChnls):
            for i in range(0,numFilters):
                for j in range(0,xlen,2):
                    for k in range(0,ylen,2):
                        print(wtMat[j,k,c,i])

                        #    wtMat[j,k,0,i] = 0#float('%.5f'%(w[j,k,0,i]))



#create a list to hold the differences
wDiffConv = []+ wOrigConv
wDiffFc = []+ wOrigFc

# function to convert the original array to all Positive


def changeOrig(wtListConv,wtListFc):
    import math
    num = len(wtListConv)


    for wtMat in wtListConv:
        shape = wtMat.shape
        xlen = shape[0]
        ylen = shape[1]
        numChnls = shape[2]
        numFilters = shape[3]

        for c in range(0,numChnls):
            for i in range(0,numFilters):
                for j in range(0,xlen):
                    for k in range(0,ylen):
                        #stepper = pow(10.0, digits)
                        #wtMat[j,k,c,i] = float('%.5f'%(wtMat[j,k,c,i]))

                        if(wtMat[j,k,c,i])<0:
                            wtMat[j,k,c,i] = -1 * wtMat[j,k,c,i]


    for wtMat in wtListFc:
        shape = wtMat.shape
        xlen = shape[0]
        ylen = shape[1]

        for i in range(0,xlen):
            j=0
            while j<ylen:
               # stepper = pow(10.0, digits)
               # wtMat[i,j]=math.trunc(stepper*wtMat[i,j]) / stepper
                if wtMat[i,j]<0:
                    wtMat[i,j] = -1 * wtMat[i,j]
                j=j+1;



# original filters are changed to all positive here
changeOrig(wOrigConv, wOrigFc)

# APPROXIMATION TECHNIQUES

def roundToX(wtListConv, wtListFc, x):
    import math
    from math import log10, floor
    num = len(wtListConv)


    for wtMat in wtListConv:
        shape = wtMat.shape
        xlen = shape[0]
        ylen = shape[1]
        numChnls = shape[2]
        numFilters = shape[3]

        for c in range(0,numChnls):
            for i in range(0,numFilters):
                for j in range(0,xlen):
                    for k in range(0,ylen):
                        #stepper = pow(10.0, digits)
                        #wtMat[j,k,c,i] = float('%.5f'%(wtMat[j,k,c,i]))
                        #num = wtMat[j,k,c,i]
                        wtMat[j,k,c,i] = round(wtMat[j,k,c,i], x -int(floor(log10(abs(wtMat[j,k,c,i])))))
                        #wtMat[j,k,c,i] = math.trunc(stepper*wtMat[j,k,c,i]) / stepper


    for wtMat in wtListFc:
        shape = wtMat.shape
        xlen = shape[0]
        ylen = shape[1]

        for i in range(0,xlen):
            j=0
            while j<ylen:
                #stepper = pow(10.0, digits)
                wtMat[i,j]=round(wtMat[i,j], x -int(floor(log10(abs(wtMat[i,j])))))
                j=j+1;

# here, we assign the approximation technique to our filters and make them positive
roundToX(wConv, wFc, 3)
changeOrig(wConv, wFc)


def assignFilters(wConv,wFc):
    for wIdx in range(0,len(weightsConvMat)):
        assign_op = weightsConvMat[wIdx].assign(wConv[wIdx])
        session.run(assign_op)

    for wIdx in range(0,len(weightsFcMat)):
        assign_op = weightsFcMat[wIdx].assign(wFc[wIdx])
        session.run(assign_op)


# In[419]:


assignFilters(wConv,wFc)


# In[41]:


def loadValues(numImg):



    feed_dict = {a: imageTest}
    values_conv1 = session.run(conv1, feed_dict=feed_dict)
    values_conv2 = session.run(conv2, feed_dict=feed_dict)
    #values_conv4 = session.run(conv4, feed_dict = feed_dict)
    values_fc1 = session.run(fc1, feed_dict = feed_dict)
    values_fc2 = session.run(fc2, feed_dict = feed_dict)
    values_fc3 = session.run(logits, feed_dict = feed_dict)
    #values_fc4 = session.run(fc4, feed_dict = feed_dict)
    #values_out = session.run(out, feed_dict = feed_dict)

    valuesConvMat = [values_conv1, values_conv2]
    valuesFcMat = [values_fc1, values_fc2, values_fc3]

    return valuesConvMat, valuesFcMat, numImg

def calculateStaticDiff(wConv, wFc, wOrigConv, wOrigFc):
    for i in range(len(wConv)):
        wDiffConv[i] = abs(wOrigConv[i]-wConv[i])
    for i in range(len(wFc)):
        wDiffFc[i] = abs(wOrigFc[i] - wFc[i])



def calculateDynamicDiff(wConv, wFc, wOrigConv, wOrigFc):
    wConvSorted = [] + wConv
    wConvOrigSorted = [] + wOrigConv
    wFcSorted = [] + wFc
    wFcOrigSorted = [] + wOrigFc
    for i in range(len(wConv)):
        wConvSorted[i] = np.sort(wConv[i])
        wConvOrigSorted[i] = np.sort(wOrigConv[i])
    for i in range(len(wFc)):
        wFcSorted[i] = np.sort(wFc[i])
        wFcOrigSorted[i] = np.sort(wOrigFc[i])
    for i in range(len(wConv)):
        wDiffConv[i] = abs(wConvOrigSorted[i] - wConvSorted[i])
        wDiffConv[i] = np.sort(wDiffConv[i])
    for i in range(len(wFc)):
        wDiffFc[i] = abs(wFcOrigSorted[i] - wFcSorted[i])
        wDiffFc[i] = np.sort(wDiffFc[i])





calculateStaticDiff(wConv, wFc, wOrigConv, wOrigFc)

def calculateConvME(valuesMat):
    shape = valuesMat.shape
    valMat = valuesMat
    numFilters = shape[3]
    means = np.zeros(numFilters)
    meanSum = np.zeros(numFilters)

    for nImg in range(0,numImg):
        meanSum = meanSum + means/numFilters
        for i in range(0,numFilters):
            result = valMat[nImg,:,:,i]
            result_sum = np.sum(result)
            means[i] = result_sum


    posME = []
    for sort in meanSum:
        if sort<0:
            sort = sort * -1
            posME.append(sort)
        else :
            posME.append(sort)



    #print(result.shape)
    average = [x / numImg for x in posME]
    averageSorted = np.sort(average)
    return average, averageSorted


# In[50]:


def calculateFCME(valuesMat):
    shape = valuesMat.shape
    valMat = valuesMat
    numFilters = shape[1]
    ##print("we are in the fully connected layer function")
    ##print(valMat)
    means = np.zeros(numFilters)
    meanSum = np.zeros(numFilters)

    for nImg in range(0,numImg):
        meanSum = meanSum + means/numFilters
        for i in range(0,numFilters):
            result = valMat[nImg,i]
            result_sum = np.sum(result)
            means[i] = result_sum

    posList = []
    for i in meanSum:
        if i<0:
            i = i*-1
        posList.append(i)
    average = [x / numImg for x in posList]
    averageSorted = np.sort(average)
    return average, averageSorted

session.run(tf.global_variables_initializer())
assignFilters(wConv,wFc)
assign_op = weightsFcMat[0].assign(wDiffFc[0])
session.run(assign_op)
assign_op = weightsConvMat[0].assign(wDiffConv[0])
session.run(assign_op)

###### LAYER 1 ######
valuesConvMat, valuesFcMat, numImg = loadValues(100)
MEConv1,MEConv1Sorted = calculateConvME(valuesConvMat[0])
MEFullyConnected1, MEFullyConnected1Sorted = calculateFCME(valuesFcMat[0])
#MEConv1 = calculateConvMEDynamic(valuesConvMat[0])
#MEFullyConnected1 = calculateFCMEDynamic(valuesFcMat[0])

assignFilters(wConv,wFc)
assign_op = weightsFcMat[1].assign(wDiffFc[1])
session.run(assign_op)
assign_op = weightsConvMat[1].assign(wDiffConv[1])
session.run(assign_op)

###### LAYER 2 ######
valuesConvMat,valuesFcMat,numImg = loadValues(100)
MEConv2,MEConv2Sorted = calculateConvME(valuesConvMat[1])
MEFullyConnected2, MEFullyConnected2Sorted = calculateFCME(valuesFcMat[1])
#MEConv2 = calculateConvMEDynamic(valuesConvMat[1])
#MEFullyConnected2 = calculateFCMEDynamic(valuesFcMat[1])


assign_op = weightsFcMat[2].assign(wDiffFc[2])
session.run(assign_op)
valuesConvMat,valuesFcMat,numImg = loadValues(100)
#MEConv1 = calculateConvMEDynamic(valuesConvMat[0])
#MEFullyConnected3 = calculateFCMEDynamic(valuesFcMat[2])
MEFullyConnected3, MEFullyConnected3Sorted = calculateFCME(valuesFcMat[2])

######### loadvalues ends here #########



MEFullyConnected1Graphs.append(MEFullyConnected1Sorted)
MEFullyConnected2Graphs.append(MEFullyConnected2Sorted)
MEFullyConnected3Graphs.append(MEFullyConnected3Sorted)
MEConv1Graphs.append(MEConv1Sorted)
MEConv2Graphs.append(MEConv2Sorted)

def plotMEConvolution(ME1, ME2):
    x1 = np.arange(len(ME1))
    x2 = np.arange(len(ME2))
    plt.bar(x1, ME1, color = 'b', align = 'center')
    d1=  len(ME1) + 1
    plt.bar(x2+d1, ME2, color = 'g', align = 'center')


# In[52]:


def plotMEFullyConnected(ME1, ME2, ME3):
    x1 = np.arange(len(ME1))
    x2 = np.arange(len(ME2))
    x3 = np.arange(len(ME3))
    plt.bar(x1, ME1, color = 'b', align = 'center')
    d1=  len(ME1) + 1
    plt.bar(x2+d1, ME2, color = 'g', align = 'center')
    d2 = d1 + len(ME2) + 1
    plt.bar(x3 + d2, ME3, color='r', align = 'center')



plotMEConvolution(MEConv1Sorted, MEConv2Sorted)
plotMEFullyConnected(MEFullyConnected1Sorted, MEFullyConnected2Sorted, MEFullyConnected3Sorted)


### copy the output to a file####
##np.savetxt('mefc1', MEFullyConnected1Graphs[0], delimiter=',')


##### dynamic diff ######


calculateDynamicDiff(wConv, wFc, wOrigConv, wOrigFc)
session.run(tf.global_variables_initializer())
assignFilters(wConv,wFc)
assign_op = weightsFcMat[0].assign(wDiffFc[0])
session.run(assign_op)
assign_op = weightsConvMat[0].assign(wDiffConv[0])
session.run(assign_op)

###### LAYER 1 ######
valuesConvMat, valuesFcMat, numImg = loadValues(100)
MEConv1,MEConv1Sorted = calculateConvME(valuesConvMat[0])
MEFullyConnected1, MEFullyConnected1Sorted = calculateFCME(valuesFcMat[0])
#MEConv1 = calculateConvMEDynamic(valuesConvMat[0])
#MEFullyConnected1 = calculateFCMEDynamic(valuesFcMat[0])

assignFilters(wConv,wFc)
assign_op = weightsFcMat[1].assign(wDiffFc[1])
session.run(assign_op)
assign_op = weightsConvMat[1].assign(wDiffConv[1])
session.run(assign_op)

###### LAYER 2 ######
valuesConvMat,valuesFcMat,numImg = loadValues(100)
MEConv2,MEConv2Sorted = calculateConvME(valuesConvMat[1])
MEFullyConnected2, MEFullyConnected2Sorted = calculateFCME(valuesFcMat[1])
#MEConv2 = calculateConvMEDynamic(valuesConvMat[1])
#MEFullyConnected2 = calculateFCMEDynamic(valuesFcMat[1])


assign_op = weightsFcMat[2].assign(wDiffFc[2])
session.run(assign_op)
valuesConvMat,valuesFcMat,numImg = loadValues(100)
#MEConv1 = calculateConvMEDynamic(valuesConvMat[0])
#MEFullyConnected3 = calculateFCMEDynamic(valuesFcMat[2])
MEFullyConnected3, MEFullyConnected3Sorted = calculateFCME(valuesFcMat[2])

######### loadvalues ends here #########



MEFullyConnected1Graphs.append(MEFullyConnected1Sorted)
MEFullyConnected2Graphs.append(MEFullyConnected2Sorted)
MEFullyConnected3Graphs.append(MEFullyConnected3Sorted)
MEConv1Graphs.append(MEConv1Sorted)
MEConv2Graphs.append(MEConv2Sorted)

##np.savetxt('mefc2', MEFullyConnected1Graphs[1], delimiter=',')
def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()


import pickle
write(MEFullyConnected1Graphs[0], "mefc1")
write(MEFullyConnected1Graphs[1], "mefc2")



def returnthis(MEFullyConnected1Graphs):
    return MEFullyConnected1Graphs

    
print("this also works for us")
