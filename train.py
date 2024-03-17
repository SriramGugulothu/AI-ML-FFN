from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
#!pip install wandb
import wandb
from wandb.keras import WandbCallback
import socket
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project ='DL_assignment')
# (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# train_images, validation_images,train_labels, validation_labels  = train_test_split(X_train,Y_train,test_size = 0.1)
def pictures():
  (train_images2, train_labels2), (test_images2, test_labels2) = fashion_mnist.load_data()
  mySet = np.unique(train_labels2)
  myData  = []
  count = 10
  for train_image,train_label in zip(train_images2,train_labels2):
    if(train_label in mySet ):
      index = np.where(mySet == train_label)
      myData.append(train_image)
      mySet = np.delete(mySet,index)
      count-=1
    if(count == 0):
      break
  names =['Ankle boot','T-shirt/top','Dress','Pullover','sneaker','Sandal','Trouser','Shirt','Coat','Bag']
  fig, axes = plt.subplots(2, 5, figsize=(5, 5))
  for i, ax in enumerate(axes.flat):
      ax.set_title(names[i])
      ax.imshow(myData[i], cmap=plt.cm.binary)
      ax.axis('off')
  plt.tight_layout()
  wandb.log({'plt':plt})
  plt.show()

def _softmax(z):
  eps = 1e-6
  return (np.exp(z-max(z)) / (sum(np.exp(z-max(z))) + eps))
def _sigmoid(z):
  z = np.clip(z, -500, 500)
  return 1/(1+np.exp(-z))
  # return sigmoid_z
def _sigmoid_derivative(z):
  return _sigmoid(z) * (1 - _sigmoid(z))
def _tanh(z):
  clipped_z = np.clip(z, -50, 50)
  return np.tanh(clipped_z)
def _tanh_derivative(z):
  return 1-np.tanh(z)**2
def _relu(z):
  return np.maximum(0,z)
def _relu_derivative(z):
  return np.where(z>0,1,0)

def _identity(z):
  return z 

def _identity_derivative(z): 
  res = np.ones(z.shape)
  return res

def backwardPropogation(w,h,a,clas,n,train_image,activFunc,lossFun):
  dw = [0 for i in range(n+1)]
  db = [0 for i in range(n+1)]
  e_l = np.zeros((10,1))
  e_l[clas] = 1  # ([0,1,0,0,0,0,0,0,0,0])
  if(lossFun == 'entropy'):
    da = -(e_l - h[n])
  else :
    da = (h[n]-e_l)*(h[n])*(1-h[n])
  layers = len(w)-1
  while(layers>0):
    dw[layers] = np.matmul(da,(h[layers-1].T))
    db[layers] = np.copy(da)
    dh_n_1 = np.matmul((w[layers].T),da) #it will be used for and in below step only
    if(activFunc == 'sigmoid'):
      da = np.multiply(dh_n_1,_sigmoid_derivative(a[layers-1])) #for next iteration
    elif(activFunc == 'tanh'):
      da = np.multiply(dh_n_1,_tanh_derivative(a[layers-1]))
    elif(activFunc == 'relu'):
      da = np.multiply(dh_n_1,_relu_derivative(a[layers-1]))
    else:
      da = np.multiply(dh_n_1,_identity_derivative(a[layers-1]))
    layers-=1
  dw[0] = np.matmul(da,train_image.T)
  db[0] = np.copy(da)
  return dw,db

def forwardPropogation(w,b,n,train_image,activFunc):
  a = [0 for i in range(n+1)]
  h = [0 for i in range(n+1)]
  for i in range(0,n):
    if(i == 0 ):
      a[i] = np.matmul(w[i], train_image) + b[i]
    else:
      a[i] = np.matmul(w[i],h[i-1]) + b[i]
    if(activFunc == 'sigmoid'):
      h[i] = _sigmoid(a[i])
    elif(activFunc == 'tanh'):
      h[i] = _tanh(a[i])
    elif(activFunc == 'relu'):
      h[i] = _relu(a[i])
    else:
      h[i] = _identity(a[i])
  a[n] =  np.matmul(w[n],h[n-1]) + b[n]
  h[n] = np.copy(_softmax(a[n]))
  return a,h

def accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun):
  a,h = forwardPropogation(w,b,n,train_image,activFunc)
  res = np.copy(h[n])
  maxi = res[0]
  label = 0
  for i in range(1,10):
    if(res[i] > maxi ):
      maxi = res[i]
      label = i
  if(lossFun == 'entropy'):
    return label , -np.log(res[train_label]+(1e-5))
  else:
    a_l = np.zeros((10,1))
    a_l[train_label] = 1
    return label, np.sum((h[n]-a_l)**2)

def validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun):
  a,h = forwardPropogation(w,b,n,validation_image,activFunc)
  res = np.copy(h[n])
  maxi = res[0]
  label = 0
  for i in range(1,10):
    if(res[i] > maxi ):
      maxi = res[i]
      label = i
  if(lossFun == 'entropy'):
    return label,-np.log(res[validation_label]+(1e-5))
  else:
    a_l = np.zeros((10,1))
    a_l[validation_label] = 1
    return label, np.sum((h[n]-a_l)**2)

def intializer(n,neurons,weightInit):
  w = []
  b = []
  prev = 784
  if(weightInit == 'random'):
    for i in range (0,n):
      w.append(np.random.randn(neurons,prev))
      b.append(np.random.randn(neurons,1))
      prev = neurons
    w.append(np.random.randn(10,prev))
    b.append(np.random.randn(10,1))  #[(100,1),(100,1),(10,1)]
    # Weights are intitialzed  [(100,784),(100,100),(10,100)]
  else:
    for i in range (0,n):
      w.append(np.random.randn(neurons,prev))
      b.append(np.zeros((neurons,1)))
      prev = neurons
    w.append(np.random.randn(10,prev))
    b.append(np.zeros((10,1)))  #[(100,1),(100,1),(10,1)]
  return w,b

def SGD(n,neurons,maxIter,weightDecay,learnRate,batchsize,weightInit,activFunc,lossFun):
  t = 0
  w,b = intializer(n,neurons,weightInit)
  # biases are initiailzed
  count = 0
  while(t < maxIter):
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      a,h = forwardPropogation(w,b,n,train_image,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image,activFunc,lossFun) # here 1 is class label
      for i in range(0,n+1):
        w[i] = w[i] - (learnRate) * ( dw[i]) - (weightDecay * w[i])
        b[i] = b[i] - (learnRate)* ( db[i])
    wandb.log({'epoch':t})
    t+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    wandb.log({'train_accuracy':count/540})
    wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc +=1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    wandb.log({'validation_accuracy':validationAcc/60})
    wandb.log({'validation_loss':total_vcE/60})

  return count

def MGD(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,momentum):
  w,b = intializer(n,neurons,weightInit)
  #ddw,ddb,pr_uw,pr_ub = init2(n,neurons)
  pr_uw = [0 for i in range(0,n+1)]
  pr_ub = [0 for i in range(0,n+1)]
  eta = learnRate
  beta = momentum
  t = 0
  while(t < maxIter):

    temp = 1
    ddw = [0 for i in range(0,n+1)]
    ddb = [0 for i in range(0,n+1)]
    for train_image,train_label in zip(train_images,train_labels):
      train_image2 = train_image.reshape(784,1) / 255
      a,h = forwardPropogation(w,b,n,train_image2,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image2,activFunc,lossFun)
      for i in range(0,n+1):
        ddw[i] += dw[i]
        ddb[i] += db[i]
      if(temp % batchSize == 0):
        for i in range(0,n+1):
          uw =  ddw[i] + (beta * pr_uw[i])
          ub =  ddb[i] + (beta * pr_ub[i])
          w[i] = (w[i] -  (eta * uw) - (weightDecay * w[i]))
          b[i] = (b[i] -  (eta * ub))
          pr_uw[i] = np.copy(uw)
          pr_ub[i] = np.copy(ub)
          for dddw in ddw:
            dddw[:] = 0
          for dddb in ddb:
            dddb[:] = 0
      temp+=1
    wandb.log({'epoch':t})
    t+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    wandb.log({'train_accuracy':count/540})
    wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc += 1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    wandb.log({'validation_accuracy':validationAcc/60})
    wandb.log({'validation_loss':total_vcE/ 6000})

  return count

def NGD(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun):
  w,b = intializer(n,neurons,weightInit) #initialization
  pr_uw = [0 for i in range(0,n+1)] #history
  pr_ub = [0 for i in range(0,n+1)]
  v_w = [0 for i in range(0,n+1)]
  v_b = [0 for i in range(0,n+1)] # these is for advance leap
  eta = learnRate
  beta = 0.9
  t = 0
  while(t < maxIter):
    temp = 1
    ddw = [0 for i in range(0,n+1)]
    ddb = [0 for i in range(0,n+1)]
    dummy_w = [0 for i in range(0,n+1)]
    dummy_b = [0 for i in range(0,n+1)]
    for i in range(0,n+1):
      w[i] = w[i] - beta * pr_uw[i]
      b[i] = b[i] - beta * pr_ub[i]
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      a,h = forwardPropogation(w,b,n,train_image,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image,activFunc,lossFun)
      for i in range(0,n+1):
        ddw[i] += dw[i]
        ddb[i] += db[i]
      if(temp % batchSize == 0):
        for i in range(0,n+1):
          w[i] = w[i]- eta * ddw[i] - (weightDecay * w[i])
          b[i] = b[i] - eta * ddb[i]
          pr_uw[i] =   eta * ddw[i] + beta * pr_uw[i]
          pr_ub[i] =   eta * ddb[i] + beta * pr_ub[i]
        for dddw in ddw:
          dddw[:] = 0
        for dddb in ddb:
          dddb[:] = 0
      temp+=1
    wandb.log({'epoch':t})
    t+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    wandb.log({'train_accuracy':count/540})
    wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc += 1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    wandb.log({'validation_accuracy':validationAcc/60})
    wandb.log({'validation_loss':total_vcE/ 6000})
  return count

def rmsProp(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta1):
  w,b = intializer(n,neurons,weightInit)
  v_w = [0 for i in range(0,n+1)]
  v_b = [0 for i in range(0,n+1)]
  eps = 1e-4
  t = 0
  beta = beta1
  eta = learnRate
  while(t < maxIter):
    temp = 1
    ddw = [0 for i in range(0,n+1)]
    ddb = [0 for i in range(0,n+1)]
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      a,h = forwardPropogation(w,b,n,train_image,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image,activFunc,lossFun)
      for i in range(0,n+1):
        ddw[i] += dw[i]
        ddb[i] += db[i]
      if(temp % batchSize == 0):
        for i in range(0,n+1):
          v_w[i] =  (1-beta)* (ddw[i] ** 2) + beta * v_w[i]
          v_b[i] =  (1-beta) * (ddb[i] ** 2) + beta * v_b[i]
          w[i] = w[i] - eta * ddw[i]/(np.sqrt(v_w[i])+eps) - (weightDecay * w[i])
          b[i] = b[i] - eta * ddb[i]/(np.sqrt(v_b[i])+eps)
          for dddw in ddw:
            dddw[:] = 0
          for dddb in ddb:
            dddb[:] =0
      temp+=1
    wandb.log({'epoch':t})
    t+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    wandb.log({'train_accuracy':count/540})
    wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc += 1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    wandb.log({'validation_accuracy':validationAcc/60})
    wandb.log({'validation_loss':total_vcE/ 6000})
  return count

def adam(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,b1,b2):
  w,b = intializer(n,neurons,weightInit)
  beta1 = b1
  beta2 = b2
  eps = 1e-10
  eta = learnRate
  m_w = [0 for i in range(0,n+1)]
  m_b = [0 for i in range(0,n+1)]
  v_w = [0 for i in range(0,n+1)]
  v_b = [0 for i in range(0,n+1)]

  for iter in range (0,maxIter):
    temp = 1
    ddw = [0 for i in range(0,n+1)]
    ddb = [0 for i in range(0,n+1)]
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1) / 255.0
      a,h = forwardPropogation(w,b,n,train_image,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image,activFunc,lossFun)
      for i in range(0,n+1):
        ddw[i] += dw[i]
        ddb[i] += db[i]
      if(temp % batchSize == 0):
        for i in range(0,n+1):
          m_w[i] = (1-beta1)*ddw[i] + beta1 * m_w[i]
          m_b[i] = (1-beta1)*ddb[i] + beta1 * m_b[i]
          v_w[i] = (1-beta2) * ddw[i]**2 + beta2 * v_w[i]
          v_b[i] = (1-beta2) * ddb[i]**2 + beta2 * v_b[i]
          #computed intermediate values
          m_w_hat = m_w[i] / (1-np.power(beta1,iter+1))
          m_b_hat = m_b[i] / (1-np.power(beta1,iter+1))
          v_w_hat = v_w[i] / (1-np.power(beta2,iter+1))
          v_b_hat = v_b[i] / (1-np.power(beta2,iter+1))
          #update parameters
          w[i] = w[i] - eta * m_w_hat / (np.sqrt(v_w_hat)+eps) - (weightDecay * w[i])
          b[i] = b[i] - eta* m_b_hat/ (np.sqrt(v_b_hat)+eps)
        for dddw in ddw:
          dddw[:] = 0
        for dddb in ddb:
          dddb[:] = 0
      temp+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    wandb.log({'train_accuracy':count/540})
    wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc += 1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    wandb.log({'validation_accuracy':validationAcc/60})
    wandb.log({'validation_loss':total_vcE/ 6000})
    wandb.log({'epoch':iter})
    #andb.log({"accuracy" : count/600})
  return count

def nadam(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,b1,b2):
  w,b = intializer(n,neurons,weightInit)
  eta = learnRate
  beta1 = b1
  beta2 = b2
  eps = 1e-10
  m_w = [0 for i in range(0,n+1)]
  m_b = [0 for i in range(0,n+1)]
  v_w = [0 for i in range(0,n+1)]
  v_b = [0 for i in range(0,n+1)]
  for iter in range(0,maxIter):
    temp = 1
    ddw = [0 for i in range(0,n+1)]
    ddb = [0 for i in range(0,n+1)]
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1) / 255.0
      a,h = forwardPropogation(w,b,n,train_image,activFunc)
      dw,db = backwardPropogation(w,h,a,train_label,n,train_image,activFunc,lossFun)
      for i in range(0,n+1):
        ddw[i] += dw[i]
        ddb[i] += db[i]
      if(temp % batchSize == 0):
        for i in range(0,n+1):
          m_w [i]= (1-beta1)*ddw[i] + beta1 * m_w[i]
          m_b[i] = (1-beta1)*ddb[i] + beta1 * m_b[i]
          v_w[i] = (1-beta2) * ddw[i]**2 + beta2 * v_w[i]
          v_b[i] = (1-beta2) * ddb[i]**2 + beta2 * v_b[i]
          #computed intermediate values
          m_w_hat = m_w[i] / (1-np.power(beta1,iter+1))
          m_b_hat = m_b[i] / (1-np.power(beta1,iter+1))
          v_w_hat = v_w[i] / (1-np.power(beta2,iter+1))
          v_b_hat = v_b[i] / (1-np.power(beta2,iter+1))
          #update parameters
          w[i] = w[i] - (eta / (np.sqrt(v_w_hat+eps)))*(beta1 * m_w_hat + (1-beta1)*ddw[i]/(1-beta1**(iter+1))) - (weightDecay * w[i])
          b[i] = b[i] - (eta / (np.sqrt(v_b_hat+eps)))*(beta1 * m_b_hat + (1-beta1)*ddb[i]/(1-beta1**(iter+1)))
        for dddw in ddw:
          dddw[:] = 0
        for dddb in ddb:
          dddb[:] = 0
      temp+=1
    count=0
    total_acE = 0
    for train_image,train_label in zip(train_images,train_labels):
      train_image = train_image.reshape(784,1)/255.0
      label , cE = accuracyCalc(train_image,train_label,w,b,n,activFunc,lossFun)
      if( label == train_label):
          count+=1
      total_acE += cE
    print("train_accuracy","train_loss",count/540,total_acE/54000)
    # wandb.log({'train_accuracy':count/540})
    # wandb.log({'train_loss':total_acE/ 54000})
    validationAcc = 0
    total_vcE = 0
    for validation_image,validation_label in zip(validation_images,validation_labels):
      validation_image = validation_image.reshape(784,1)/255.0
      label_v,vE = validationAccuracy(validation_image,validation_label,w,b,n,activFunc,lossFun)
      if( label_v == validation_label):
        validationAcc += 1
      total_vcE += vE
    print("validation_accuracy","validation_loss",validationAcc/60,total_vcE/6000)
    # wandb.log({'validation_accuracy':validationAcc/60})
    # wandb.log({'validation_loss':total_vcE/ 6000})
    # wandb.log({'epoch':iter})
  return w,b

def confusionMatrix(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta1,beta2):
  w,b = nadam(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta1,beta2)
  predicted = []
  truth = []
  count = 0
  for X,Y in zip(X_test,Y_test):
    test_image = X.reshape(784,1)/255.0
    label , cE = accuracyCalc(test_image,Y,w,b,n,activFunc,lossFun)
    if(label == Y):
      count+=1
    predicted.append(label)
    truth.append(Y)
  print(count/100)
  cm = confusion_matrix(truth,predicted)
  plt.figure(figsize=(8, 6))
  sn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix4')
  plt.savefig('confusion_matrix4.png')
  # Log the saved image using wandb
  wandb.log({"confusion_matrix4": wandb.Image('confusion_matrix4.png')})
  # Show the plot
  plt.show()

def mainFunction(maxIter,n,neurons,weightDecay,learnRate,optimizer,batchSize,weightInit,activFunc,lossFun,momentum,beta,beta1,beta2,epsilon):
  if(optimizer == 'sgd'):
    accuracy = SGD(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun)
  elif(optimizer == 'mgd'):
    accuracy = MGD(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,momentum)
  elif(optimizer == 'ngd'):
    accuracy = NGD(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun)
  elif(optimizer == 'rmsProp'):
    accuracy = rmsProp(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta)
  elif(optimizer == 'adam'):
    accuracy = adam(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta1,beta2)
  else:
    w,b = nadam(n,neurons,maxIter,weightDecay,learnRate,batchSize,weightInit,activFunc,lossFun,beta1,beta2)
    #confusionMatrix(3,128,5,0,0.001,64,'Xavier','sigmoid','entropy',0.9,0.99) we can use this for confusion matrix

def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment',
                        help='Project name used to track experiments in Weights & Biases dashboard')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='dlassignment',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
  
  parser.add_argument('-d', '--dataset', type=str, default='mnist',choices=["mnist", "fashion_mnist"],
                        help='Dataset choice: ["mnist", "fashion_mnist"]')
  
  parser.add_argument('-e', '--epochs', type=int, default=5,help='Number of epochs to train neural network')

  parser.add_argument('-b', '--batch_size', type=int, default=64,help='Batch size to train neural network')

  parser.add_argument('-l', '--loss', type=str, default='entropy',choices=["mean_squared_error", "cross_entropy"],help='Choice of MSE or Entropy')
  
  parser.add_argument('-o', '--optimizer', type=str, default='rmsProp', choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0.001, help='Learning rate used to optimize model parameters')

  parser.add_argument('--momentum', '-m', type=int, default=0.5, help='Momentum used by momentum and nag optimizers')

  parser.add_argument('--beta', '-beta', type=int, default=0.5, help='Beta used by rmsprop optimizer')

  parser.add_argument('--beta1', '-beta1', type=int, default=0.9, help='Beta1 used by adam and nadam optimizers')

  parser.add_argument('--beta2', '-beta2', type=int, default=0.99, help='Beta2 used by adam and nadam optimizers')

  parser.add_argument('--epsilon', '-eps', type=int, default=0.000001, help='Epsilon used by optimizers')

  parser.add_argument('--weight_init', '-w_i', type=str, default='Xavier',choices=["random", "Xavier"], help='random used by optimizers')

  parser.add_argument('--weight_decay', '-w_d', type=int, default=0.0005, help='Weight decay used by optimizers')

  parser.add_argument('--num_layers', '-nhl', type=int, default=3, help='Number of hidden layers used in feedforward neural network')
  
  parser.add_argument('--hidden_size', '-sz', type=int, default=128, help='Number of hidden neurons in a feedforward layer')

  parser.add_argument('--activation', '-a', type=str, default='tanh',choices=["identity", "sigmoid", "tanh", "ReLU"], help='Number of hidden neurons in a feedforward layer')

  return parser.parse_args()

args = parse_arguments()

wandb.init(project=args.wandb_project)

if args.dataset == 'mnist':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    train_images, validation_images,train_labels, validation_labels  = train_test_split(X_train,Y_train,test_size = 0.1)
else:
  (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
  train_images, validation_images,train_labels, validation_labels  = train_test_split(X_train,Y_train,test_size = 0.1)
  
wandb.run.name=f'activation {args.activation} weight_init{args.weight_init}opt{args.optimizer}'

mainFunction(args.epochs,args.num_layers,args.hidden_size,args.weight_decay,args.learning_rate,args.optimizer,args.batch_size,args.weight_init,args.activation,args.loss,args.momentum,args.beta,args.beta1,args.beta2,args.epsilon)
