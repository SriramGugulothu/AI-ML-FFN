# DL_assignment.ipynb
This document is helpful in understanding the code implementation

The code is structured as follows:
**Data**:
    First I have downloaded the dataset
    I have used train_test_split method to split the data into training data and validation data (10%)
 **Libraries**:
    - I have imported the necesary libraries numpy, wandb, matplotlib,sklearn and seaborn
** Activation Function**:
    - I have written activation functions and their derivatives respectively.
    - They take input as numpy array, do operations and returns the numpy array based on activation function
 **Weight and Bias intialisation**:
    - I have used Initializer() function to generate the weights and biases depending on weights initialization and
      fixing their shapes here depending on the neurons in hidden layer 
 **ForwardPropogation**:
   - It take weights, biases and training image.
   - It returns the activation and pre-activation values
 **BackwardPropogation**:
   - It takes the weights , preactivation and activation to generate gradients of weights and biases depending on the
     loss function passed.
**accuracyCalc**:
  - It takes the weights , biases, activFunc
  - It calls the ForwardPropogation function(It gives softmax). It returns the maximum probability label and either cross entropy of label or mean squared error
validationAccuracy:
- It does the same thing as accuracyCalc but it is for validation data.
**Optimization Algorithms**
  Every function takes this parameters = n,neurons,maxIter,weightDecay,learnRate,batchsize,weightInit,activFunc,lossFuc
  n- number of hidden layers
  neurons - number of nuerons in each layer
  maxIter- number of epochs
  for other parameters names depicts their association.
  -In each algorithm, I accumulate the summation of gradients of both weights and biases and update the weights and biases  
  after reaching batch size.
  - For the shake of plotting, I am calculating the training accuracy, validation accuracy, train loss and validation loss for each epoch and ploting them using wandb.
**confusionMatrix**:
 - I have used this function for constructing the confusion matrix.
 - Inside that I have used nadam, becuase it gave me the better validation accuracy
WandbConnection:
- Last cell is dedicated for wandb connection. 

