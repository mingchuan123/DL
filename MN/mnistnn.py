
# coding: utf-8

# In[677]:

import numpy as np
from data_utils import DataUtils

trainfile_X = 'train-images-idx3-ubyte'
trainfile_y = 'train-labels-idx1-ubyte'
testfile_X = 't10k-images-idx3-ubyte'
testfile_y = 't10k-labels-idx1-ubyte'
        
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()
print(train_y[0:10])


# In[678]:

def normalize(x):
    
    encoding = np.zeros((len(x),10))
    encoding[np.arange(len(x)),x] = 1
    return encoding


# In[679]:

train_y = normalize(train_y)
print(train_y[0:10])
test_y = normalize(test_y)
print(type(test_y))


# In[650]:

def sigmoid(x):
    
    return 1/(1 + np.exp(-x))

def prime_sigmoid(x):
    
    return sigmoid(x) * (1 - sigmoid(x))

def layer(weights,x,bias):
    
    return np.add(np.dot(weights,x),bias)
    
def create_weights(mu,sigma,dimension):
    np.random.seed(100)
    weights = np.random.normal(mu,sigma,dimension[0]*dimension[1])
    return weights.reshape(dimension[0],dimension[1])
    
def soft_max(x):
    
    return np.exp(x)/sum(np.exp(x))


# In[651]:

mu = 0.0
sigma = 0.1
image_d = 784
neuron_d = 30
neuron_d_output = 10
learnrate = 0.1

dimension_1 = [neuron_d,image_d]
weights_1 = create_weights(mu,sigma,dimension_1)
bias_1 = np.zeros(neuron_d)
dimension_2 = [neuron_d,neuron_d]
weights_2 = create_weights(mu,sigma,dimension_2)
bias_2 = np.zeros(neuron_d)

dimension_3 = [neuron_d_output,neuron_d]
weights_3 = create_weights(mu,sigma,dimension_3)
bias_3 = np.zeros(neuron_d_output)


# In[652]:

def forward(x):
    
    global weights_1
    global weights_2
    global weights_3
    global bias_1
    global bias_2
    global bias_3
    
    #forward
    layer_1_i = layer(weights_1,x,bias_1)
    layer_1_o = sigmoid(layer_1_i)
    layer_2_i = layer(weights_2,layer_1_o,bias_2)
    layer_2_o = sigmoid(layer_2_i)
    layer_3_i = layer(weights_3,layer_2_o,bias_3)
    layer_3_o = sigmoid(layer_3_i)
    layer_i = np.array([layer_1_i,layer_2_i,layer_3_i])
    layer_o = np.array([layer_1_o,layer_2_o,layer_3_o])
     
    return layer_i,layer_o
    

def backward(x,y,layer_i,layer_o):
    
    global weights_1
    global weights_2
    global weights_3
    global bias_1
    global bias_2
    global bias_3 
    
    layer_1_i = layer_i[0]
    layer_2_i = layer_i[1]
    layer_3_i = layer_i[2]
    layer_1_o = layer_o[0]
    layer_2_o = layer_o[1]
    layer_3_o = layer_o[2]
    
    #backward
    loss = np.subtract(y,layer_3_o)
    layer_b_3 = np.multiply(loss,prime_sigmoid(layer_3_i))
    layer_b_3_w = np.outer(layer_b_3,layer_2_o)

    layer_b_2 = np.multiply(np.dot(weights_3.T,layer_b_3),prime_sigmoid(layer_2_i))
    layer_b_2_w = np.outer(layer_b_2,layer_1_o)

    layer_b_1 = np.multiply(np.dot(weights_2.T,layer_b_2),prime_sigmoid(layer_1_i))
    layer_b_1_w = np.outer(layer_b_1,x)
    
    layer_b_w = np.array([layer_b_3_w,layer_b_2_w,layer_b_1_w])
    layer_b = np.array([layer_b_3,layer_b_2,layer_b_1])
    
    return layer_b_w,layer_b
    


# In[654]:

epoches = 2
batch_size = 3000
batches = int(len(train_X)/batch_size)
total = 0

for epoch in range(epoches):
    for i in range(batches):
        loss = 0
        accuracy = 0
        start = i * batch_size
        end = (i+1) * batch_size
        for j in range(start,end):
            x = train_X[j]
            y = train_y[j]
            
            layer_i,layer_o = forward(x)
            layer_b_w,layer_b = backward(x,y,layer_i,layer_o)

            weights_3 = np.add(weights_3,learnrate * layer_b_w[0])
            weights_2 = np.add(weights_2,learnrate * layer_b_w[1])
            weights_1 = np.add(weights_1,learnrate * layer_b_w[2])
            bias_3 = np.add(bias_3,learnrate * layer_b[0])
            bias_2 = np.add(bias_2,learnrate * layer_b[1])
            bias_1 = np.add(bias_1,learnrate * layer_b[2])
            
            loss += sum(abs(np.subtract(y,layer_o[2])))
            accuracy += int(layer_o[2].argmax() == y.argmax())
        accuracy = accuracy/batch_size
        total += accuracy
        print('batch:{} , loss:{}, accuracy:{}'.format(i,loss,accuracy))
        
        
#print('total accuracy:{}'.format(total))


# In[689]:

accuracy = 0
for i in range(len(test_y)):
    _,output = forward(test_X[i])
    accuracy += int(output[2].argmax() == test_y[i].argmax())
print('test accuracy:{}'.format(float(accuracy/len(test_y))))


# In[ ]:




# In[ ]:




# In[ ]:



