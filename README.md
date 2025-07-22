

#FeedForward.ipynb

This document is helpful in understanding the code implementation **FNN.ipynb**
The code is structured as follows:
**DataSet**:
    - I have downloaded the fashion_mnist dataset
    - I have used train_test_split library to split the dataset into training data (90%) and 
      validation data (10%)
 **Libraries**:
    - I have imported the necesary libraries numpy, wandb, matplotlib,sklearn and seaborn
**Printing Labels**:
    - I have acquired unique labels and plotted them accordingly.
    - I have commented the wandb.log() here. It can be uncommented to replot the images 
      in wandb
**Activation Functions**:
    - I have written activation functions such as sigmoid, tanh, relu and identity and 
      their derivatives respectively.
    - For both activation fucntions and derivative functions parameter is pre-activation 
      numpy array with shape (size of hidden layer, 1) 
    - list a = [], is used to store the pre_activations and h = [] to store the
     activation values at each layer.
     
 **Weight and Bias intialisation**:
    - I have used Initializer() function to generate the weights and biases depending on 
      weights initialization parameter - random or xavier.
    - It also fixes the shapes of weights and biases
 **ForwardPropogation**:
   - It take weights, biases and training image(which is reshaped into (784,1) numpy array) and number of hidden layers.
   - n denotes number of hidden layers
   - It returns the list of activation and pre-activation values 
 **BackwardPropogation**:
   - It takes the weights(w) , activation (h[]) and pre-activation (a[]) values to calculate the gradients of weights and biases.
  - It returns the list of weights gradients (dw) and biases gradients (db)
**accuracyCalc**:
  - It takes the weights(w) , biases (b), activFunction, lossFunction, train_image and trian_label
  - It calls the ForwardPropogation function(It gives softmax in h[n]). It returns the maximum probability label along with cross entropy value or mean squared error in one go.
    
**validationAccuracy**:

- It does the same thing as accuracyCalc but it is for validation data.

**Optimization Algorithms**

  Every function takes these parameters = n,neurons,maxIter,weightDecay,learnRate,batchsize,weightInit,activFunc,lossFun
  n- number of hidden layers
  neurons - number of nuerons in each layer
  maxIter- number of epochs
  weightInit- weight initialization (random or xavier)
  
  for other parameters names depicts their association.
  
  -In each algorithm, I accumulate the summation of gradients of both weights and biases and update the weights and biases  
  after reaching batch size.
  
  - For the shake of plotting, I am calculating the training accuracy, validation accuracy, train loss(cross_entropy loss) and validation loss(cross_entropy loss) for each epoch and ploting them using wandb.
    
- Except 'nadam' each optimizer function returns the count which is the number of correctly labeled training data points after last epoch.
  
- 'nadam' return w,b which I have used for confusion matrix plotting because nadam gives the good accuracies according to my observations.
  
**confusionMatrix**:
 - I have used this function for constructing the confusion matrix.
 - Inside that I have used nadam, becuase it gave me the better validation accuracy

**Wandb**:
- Last cell is dedicated for wandb sweep runs.
- Here I forgot to add all variations of parameters in the sweep_params at the time of submission. I am sorry for this.

 # train.py
 - It is completely similar to FNN.ipynb python code.
 - I have removed wandb sweeps code.
 - Added few lines of parsers code for implementation of train.py
 - I have changed little bit of DL_assignment.ipynb python code to facilitate more parameters asked in final 
   question.
 - Here, please note that I have used these notations-
 - 'entropy' for cross_entorpy
 - 'mse' for mean_squared_error
 - 'sgd' for stochastic gradient descent
 - 'mgd' for Momentum-based gradient descent
 - 'ngd' for Nesterov Accelerated gradient descent
 - 'rmsProp' for RMSprop algorithm
 - 'adam' for adam algorithm
 - 'nadam' for nadam algorithm
# wandb_repot
 - please note that validation and accuracy losses plotted are Cross_Entropy losses.


