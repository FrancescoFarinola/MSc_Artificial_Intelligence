import numpy as np
import sys


class Layer:
    '''class Layer
        ...
        Attributes:
        -----------
        - hidden_units : int
            the number of neurons in a layer
        - activation : str
            the activation function of the layer
        - weights : np.array of size [input, hidden_units] 
            weights of the current layer
        - bias : np.array of size [1, hidden_units]
            biases of the current layer
        - input : np.array 
            contains a copy of the values that will be 'activated'
        - Z : np.array
            contains the values of the neurons
        - A : np.array 
            contains the values of neurons after the applying the activation function

        Methods:
        --------
        - __init__(hidden_units, activation): 
            Initializes attributes of the class
        - initialize_params(input_size, hidden_units)
            Initialize weights with random values and sets biases to 0
        - activate(X):
            Updates the value of neurons (Z) with weights and biases and applies the activation function
        - activation_function(z, derivative=False):
            Contains all the activation functions with their respective derivatives
            '''
    def __init__(self, hidden_units, activation=None):
        ''' Initialize attributes of the class'''
        self.hidden_units = hidden_units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input = None
        
    def initialize_params(self, input_size, hidden_units):
        '''Called inside 'activate' method if weights and bias are not yet initialized'''
        self.weights = np.random.randn(input_size, hidden_units) * np.sqrt(2/input_size) 
        self.bias = np.zeros((1, hidden_units))
    
    def activate(self, X):
        '''Makes a copy of the input; If weights are not yet initialized calls 'initialize_params' method
           Updates the value of the neuron with weights and bias
           If an activation_function is specified updates the value of the neuron by activating it
           Otherwise returns the linear value'''
        self.input = np.array(X, copy=True)
        if self.weights is None:
            self.initialize_params(self.input.shape[-1], self.hidden_units)

        self.Z = np.dot(X, self.weights) + self.bias
        
        if self.activation is not None:
            self.A = self.activation_function(self.Z)
            return self.A
        return self.Z


    def activation_function(self, z, derivative=False):
        '''
        Activation functions implemented:
            Layer activation functions: ReLU, ELU, SELU, LeakyReLU, tanh 
            Output functions: sigmoid, softmax, linear
        TODO softplus, softsign    
        '''
        if self.activation == 'linear':
            if derivative:
                return 1
            return z
        if self.activation == 'relu':
            if derivative:
                return np.where(z<0, 0, 1)
            return np.maximum(0, z)
        if self.activation == 'elu':
            alpha = 1.0
            if derivative:
                return np.where(z>=0, 1, np.where(z>=0, z, alpha * (np.exp(z) - 1) + alpha))
            return np.where(z>=0, z, alpha * (np.exp(z) - 1))
        if self.activation == 'selu':
            alpha = 1.67326
            scale = 1.0507
            if derivative:
                return np.where(z>=0, 1 * scale , np.where(z>=0, z, alpha * (np.exp(z) - 1) + alpha) * scale)
            return np.where(z>=0, z * scale , alpha * (np.exp(z) - 1) * scale)
        if self.activation == 'leakyrelu':
            alpha = 0.3
            if derivative:
                return np.where(z<0, alpha, 1)
            return np.maximum(z, alpha * z)
        if self.activation == 'sigmoid':
            if derivative:
                s = 1 / (1 + np.exp(-z))
                return s * (1 - s)
            return 1 / (1 + np.exp(-z))
        if self.activation == 'tanh':
            if derivative:
                return 1.0 - np.tanh(z) ** 2
            return np.tanh(z)
        if self.activation == 'softmax':
            if derivative: 
                exps = np.exp(z - np.max(z, axis=1, keepdims=True)) 
                return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
            exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

class EarlyStopping:
    '''class EarlyStopping
       Attributes:
       -----------
       - delta : float32
            Value of minimum improvement for satisfying the stopping criteria
       - patience : int
            Number of epochs to consider for satisfying the stopping criteria
       - restore_weights : bool
            Default = True : restores the best weights and biases of the last 
            'patience' epochs
            False : uses the last epoch's weights and biases

       Methods:
       --------
       - __init__(patience, delta, restore_weights):
            Initialize attributes with the specified values
       - calculate_stop(val_loss, loss):
            Checks whether last epoch loss plus the minimum improvement is
            lesser/greater than the last 'patience' epochs
    '''
    def __init__(self, patience=5, delta=1e-3, restore_weights=True):
        '''Initializes the attributes of the object with values specified by arguments'''
        self.delta = delta
        self.patience = patience
        self.restore_weights = restore_weights
    
    def calculate_stop(self, val_loss, loss):
        '''This function is called at the end of each epoch and checks whether 
        the stopping criteria is satisfied. Uses the variable 'loss' to
        determine whether we are calculating the stop for the accuracy metric
        (classification task) or for Mean Squared Error (regression task)
        '''
        #Store only the last 'patience' validation losses
        interval = val_loss[-self.patience:]
        stop = False
        #Classification/Accuracy case
        if loss == 'categorical_cross_entropy' or loss == 'binary_cross_entropy':
            #if last epoch loss + minimum improvement is lesser than the last 'patience' accuracy then stop
            if all(interval[-1] + self.delta < np.array(interval[:-1])):
                stop = True
        #Regression/MSE case
        if loss == 'mse':
            #if last epoch loss + minimum improvement is greater than the last 'patience' mse then stop
            if all(interval[-1] + self.delta > np.array(interval[:-1])): 
                stop = True
        return stop


class NeuralNetwork:
    '''class Neural Network
        Attributes:
        -----------
        - layers : dict of (str, Layer objects)
            Contains the Layer objects
        - nablas : dict of (str, np.array(float32))
            Contains the gradients 
        - tmp : dict of (str, np.array(float32))
            Contains the temporary values relative to current 
            epoch and necessary for various calculations
        - epochs : int
            Counter for epoch number
        - learning_rate : float32 
            Value to determine the step size to update weights and biases
        - loss : str
            Loss function to be used, depends on ML task
            MSE for regression, Binary Cross Entropy for binary classification
            Categorical Cross Entropy for multiclass classification
        - optimizer : str
            Specifies the kind of optimizer to use, to update weights and biases
            Default: None - Applies Minibatch Gradient Descent
        - earlystop : EarlyStopping object
            If None Early stop is not applied
            If specified applies Early Stop
        - restore : dict of (str, np.array(float32))
            Used only if early stop is applied. Contains weights and biases for
            each layer of the last 'patience' epochs ('patience' is an attribute
            of EarlyStopping object)
        - history : dict of (str, np.array(float32))
            Contains loss, training accuracy/mse and validation accuracy/mse

        Methods:
        --------
        - __init__ : 
            Initializes dictionaries
        - add(layer) : 
            Adds a Layer object to the layers dict
        - compile(epochs, learning_rate, loss, optimizer, earlystop)
            Initializes all attributes passed as arguments
            If optimizer and earlystop are not specified they will not be applied
        - forward(x) : 
            Cycles through each layer to update the value of neurons, activate them,
            and store the temporary values in the tmp dict
        - backprop(y) : 
            Cycles through reversed layers and calculates the gradients 
            for each layer. Better explanation in inline comments
        - update_weights(epoch, step):
            Depending on the optimizer, updates weights and biases of the
            neural network
        - fit(x_train, y_train, x_validation, y_validation, batch_size):
            Performs the fit of the neural network. Better explanation in inline 
            comments
        - loss_function(y_true, y_pred, derivative=False):
            Contains the loss functions and their derivatives. Derivatives are used
            to calculate the gradient of the last layer while the functions are used
            to calculate the loss over each epoch.
            Implemented losses: mse, binary_cross_entropy, categorical_cross_entropy
            NB: categorical_cross_entropy requires one-hot encoded labels
        - shuffle(x, y):
            Consistent method to shuffle  two lists/np.arrays of the same length 
            using the same index list
        - get_batches(x, y, batch_size):
            Shuffles the dataset and returns a tuple (batch_x, batch_y) of batches 
            with aligned indexes of size = batch_size
    '''
    def __init__(self):
        '''Initialize dictionaries'''
        self.layers = dict() 
        self.nablas = dict() 
        self.tmp = dict() 
        
    def add(self, layer):
        '''Adds a Layer object to the Neural Network'''
        self.layers[len(self.layers)+1] = layer
        
    def compile(self, epochs, learning_rate, loss, optimizer=None, earlystop=None):
        '''Initializes all parameters to be specified as argument to a specific ML task'''
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.earlystop = earlystop
        # If we want to restore weights of the epoch with the best loss
        if self.earlystop is not None and self.earlystop.restore_weights:
            self.restore = dict()
        #Initialize parameters depending on the optimizer type
        if self.optimizer is not None:
            if self.optimizer == 'momentum' or self.optimizer == 'nesterov':
                self.mu_0 = 0.5 #Starting value for mu
                self.mu_max = 0.99 #Max value for mu
                self.mom = dict() #Dictionary for momentum terms
                for idx in self.layers.keys():
                    self.mom['W'+str(idx)] = 0
                    self.mom['b'+str(idx)] = 0
            if self.optimizer == 'rmsprop':
                self.beta = 0.9 #Multiplier for momentum terms
                self.epsilon = 1e-8 #Epsilon is used to clip values, to prevent overflow/underflow
                self.mom = dict() #Dictionary for momentum terms
                for idx in self.layers.keys():
                    self.mom['W'+str(idx)] = 0
                    self.mom['b'+str(idx)] = 0
            if self.optimizer == 'adam' or self.optimizer == 'nadam':
                self.beta1 = 0.9 # Multiplier for first moment vector
                self.beta2 = 0.999 # Multiplier for second moment vector
                self.epsilon = 1e-9 #To prevent overflow/underflow
                #Adam and Nadam use two moment vectors
                self.m = dict()   
                self.v = dict()  
                for idx in self.layers.keys():
                    self.m['W'+str(idx)] = 0
                    self.m['b'+str(idx)] = 0
                    self.v['W'+str(idx)] = 0
                    self.v['b'+str(idx)] = 0 
                

    def forward(self, x):
        '''Cycles through each layer to update the value of neurons, activate them,
            and store the temporary values in the tmp dict'''
        for idx, layer in self.layers.items():
            x = layer.activate(x) # Update the value of the neuron and activate it
            #Update temporary variables mainly used in backprop function
            self.tmp['W'+str(idx)] = layer.weights
            self.tmp['b'+str(idx)] = layer.bias
            self.tmp['Z'+str(idx)] = layer.Z
            self.tmp['A'+str(idx)] = layer.A
        return x

    def backprop(self, y):
        '''Cycles through reversed layers and calculates the gradients for each layer
           i.e. Gradient calculation of last layer:
           To update weights we need to calculate the partial derivative of the cost 
           w.r.t. the weights. This can be split into 3 calculations:
                (1) partial derivative of cost w.r.t output:
                    which is the derivative of the cost function
                (2) partial derivative of output w.r.t. input:
                    considering the only function that transforms input into output
                    is the activation function, then we need to calculate only the
                    derivative of activation function 
                (3) partial derivative of input w.r.t. weights:
                    since the input of the neuron is (output of previous layer * 
                    weight + bias the derivative is simply the output of the 
                    previous layer
            A more stable and simpler computation of last layer backpropagation 
            combines step 1 and 2 to prevent overflow: prediction - target
            
            Same goes for the computation of gradients in hidden layers except
            for step 1 where we need to split the partial derivative of cost into
            two members: weight of previous layer and gradient of previous layer
            
            Computation of bias gradient is simpler since the partial derivative
            of input w.r.t to bias (output of previous layer * weight + bias) is 1,
            so as we just ignore the (3)rd step'''
            
        last_layer_idx = max(self.layers.keys())
        m = y.shape[0]
        for idx in reversed(range(1, last_layer_idx+1)): #Reverse layers
            #If last layer, gradient of Z is the cost function
            if idx == last_layer_idx:
                if self.loss == 'mse' or self.loss == 'binary_cross_entropy':
                    y = y[:, np.newaxis] #makes the target a 2D array with dimension [x, 1] in regression and binary classification case
                self.nablas['dZ'+str(idx)] = self.loss_function(y, self.tmp['A'+str(idx)], derivative=True) #(1 and 2)
            #Else compute the gradient Z of the current layer: dZn = dZ(n+1) @ W(n+1) * Inverse_Activation_Function of Zn
            else:
                self.nablas['dZ'+str(idx)] = np.dot(self.nablas['dZ'+str(idx+1)], self.tmp['W'+str(idx+1)].T) \
                                                * self.layers[idx].activation_function(self.tmp['Z'+str(idx)], derivative=True) 
            #Compute gradients
            self.nablas['dW'+str(idx)] = (1 / m) * np.dot(self.layers[idx].input.T, self.nablas['dZ'+str(idx)]) #(3)
            self.nablas['db'+str(idx)] = (1 / m) * np.sum(self.nablas['dZ'+str(idx)], axis=0, keepdims=True)

    def update_weights(self, epoch, step):
        '''This methods updates weights and biases depending on the optimizer type
           Paper with all optimizers explained: https://arxiv.org/pdf/1609.04747.pdf ''' 
        for idx in self.layers.keys():
            if self.optimizer is None or self.optimizer == 'sgd':
                #SGD optimizer simply updates weights and bias by subtracting to the current one, gradients multiplied by learning rate.
                self.layers[idx].weights -= self.learning_rate * self.nablas['dW'+str(idx)]
                self.layers[idx].bias -= self.learning_rate * self.nablas['db'+str(idx)]
            '''Those implemented are alternative computations of momentum and nesterov optimizers
               Ones referred in previous paper are not compatible with my implemented neuralnet
               The following thesis shows an alternative demonstration of Nesterov optimizer:
               Chapter 7, formulas 7.1, 7.2: https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
               Since it is an extension of simple Momentum optimizer, they mostly share the same implementation
               '''
            if self.optimizer == 'momentum':
                #Increase mu by 20% at each epoch until mu_max is reached
                # mu = 0.9
                mu = min(self.mu_0 * 1.2 ** (epoch - 1), self.mu_max)
                #Update momentum terms
                self.mom['W'+str(idx)] = mu * self.mom['W'+str(idx)]  - self.learning_rate * self.nablas['dW'+str(idx)]
                self.mom['b'+str(idx)] = mu * self.mom['b'+str(idx)]  - self.learning_rate * self.nablas['db'+str(idx)]
                #Update weights and bias
                self.layers[idx].weights += self.mom['W'+str(idx)]
                self.layers[idx].bias += self.mom['b'+str(idx)]

            if self.optimizer == 'nesterov':
                '''Section 3.5 formulas 6,7: https://arxiv.org/pdf/1212.0901v2.pdf'''
                # mu = 0.9
                mu = min(self.mu_0 * 1.2 ** (epoch - 1), self.mu_max)
                #Get previous momentum terms
                mW_prev =  np.array(self.mom['W'+str(idx)], copy=True)
                mb_prev = np.array(self.mom['b'+str(idx)], copy=True)
                #Update momentum terms
                self.mom['W'+str(idx)] = mu * self.mom['W'+str(idx)]  - self.learning_rate * self.nablas['dW'+str(idx)]
                self.mom['b'+str(idx)] = mu * self.mom['b'+str(idx)]  - self.learning_rate * self.nablas['db'+str(idx)]
                #Update weights and bias
                self.layers[idx].weights += (-mu * mW_prev + (1 + mu) * self.mom['W'+str(idx)])
                self.layers[idx].bias += (-mu * mb_prev + (1 + mu) * self.mom['b'+str(idx)])
            
            if self.optimizer == 'rmsprop':
                ''' In the following article it explains how RMSProp updates weights and bias:
                    https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b'''
                #Update moment terms
                self.mom['W'+str(idx)] = self.beta * self.mom['W'+str(idx)] + (1 - self.beta) * np.power(self.nablas['dW'+str(idx)], 2) 
                self.mom['b'+str(idx)] = self.beta * self.mom['b'+str(idx)] + (1 - self.beta) * np.power(self.nablas['db'+str(idx)], 2)
                #Update weights and bias
                self.layers[idx].weights += -self.learning_rate * self.nablas['dW'+str(idx)] / (np.sqrt(self.mom['W'+str(idx)] + self.epsilon))
                self.layers[idx].bias += -self.learning_rate * self.nablas['db'+str(idx)] / (np.sqrt(self.mom['b'+str(idx)]+ self.epsilon))

            if self.optimizer == 'adam' or self.optimizer == 'nadam':
                ''' Adam paper: https://arxiv.org/pdf/1412.6980.pdf also: https://paperswithcode.com/method/adam
                    Nadam: https://paperswithcode.com/method/nadam'''
                #Calculate the two moment vectors for both weight and bias
                self.m['W'+str(idx)] = self.beta1 * self.m['W'+str(idx)] + (1 - self.beta1) * self.nablas['dW'+str(idx)]
                self.v['W'+str(idx)] = self.beta2 * self.v['W'+str(idx)] + (1 - self.beta2) * np.power(self.nablas['dW'+str(idx)], 2) 
                
                self.m['b'+str(idx)] = self.beta1 * self.m['b'+str(idx)] + (1 - self.beta1) * self.nablas['db'+str(idx)]
                self.v['b'+str(idx)] = self.beta2 * self.v['b'+str(idx)] + (1 - self.beta2) * np.power(self.nablas['db'+str(idx)], 2)
                #Take the step/batch progress of the epoch into account
                m_step_weight  = self.m['W'+str(idx)] / (1 - np.power(self.beta1, step))
                v_step_weight = self.v['W'+str(idx)] / (1 - np.power(self.beta2, step))
                m_step_bias  = self.m['b'+str(idx)] / (1 - np.power(self.beta1, step))
                v_step_bias = self.v['b'+str(idx)] / (1 - np.power(self.beta2, step))
                #Adam weight and bias update
                if self.optimizer == 'adam':
                    self.layers[idx].weights += -self.learning_rate * m_step_weight / (np.sqrt(v_step_weight) + self.epsilon)
                    self.layers[idx].bias += -self.learning_rate * m_step_bias / (np.sqrt(v_step_bias) + self.epsilon)
                #Nadam weight and bias update
                else:
                    self.layers[idx].weights += - self.learning_rate / (np.sqrt(v_step_weight) + self.epsilon) *\
                                                     (self.beta1 * m_step_weight + (1 - self.beta1) *  self.nablas['dW'+str(idx)] / (1 - np.power(self.beta1, step)))
                    self.layers[idx].bias += - self.learning_rate / (np.sqrt(v_step_bias) + self.epsilon) *\
                                                     (self.beta1 * m_step_bias + (1 - self.beta1) *  self.nablas['db'+str(idx)] / (1 - np.power(self.beta1, step)))


    def fit(self, x_train, y_train, x_validation=None, y_validation=None, batch_size=32):
        '''Performs the fit of the neural network.'''
        losses = [] 
        train_accs = []
        val_accs = []
        for epoch in range(1, self.epochs+1):
            step = 0
            print(f'Epoch {epoch}')
            batches = self.get_batches(x_train, y_train, batch_size) #Shuffle dataset and get batches
            epoch_loss = [] #Stores losses for each batch

            for x, y in batches:
                '''NB really important that the step increase is here since Adam/Nadam optimizers 
                use the current timestep to update weights and biases'''
                step += 1 
                preds = self.forward(x) #Updates values of neurons and activate them   
                loss = self.loss_function(y, preds) #Calculate loss for current batch
                epoch_loss.append(loss)
                self.backprop(y) #Compute gradients
                self.update_weights(epoch, step) #Updates weights and biases depending on optimizer

                #Progress bar
                j = (step) / len(batches)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
                sys.stdout.flush()
            
            #Calculate total loss over the current epoch
            loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(loss)

            # If Regression case we calculate only MSE 
            if self.loss == 'mse':
                #Calculate train loss
                train_p = self.forward(x_train)
                train_loss = self.loss_function(y_train, train_p)
                train_accs.append(train_loss)
                #Calculate validation loss
                if x_validation is not None:
                    val_p = self.forward(x_validation)
                    val_loss = self.loss_function(y_validation, val_p)
                    val_accs.append(val_loss)
                #Round losses
                train_acc = np.round(sum(train_accs) / len(train_accs), 5)
                val_acc = np.round(sum(val_accs) / len(val_accs), 5)
                print(f' - Train loss: {train_acc} \t  Validation loss: {val_acc}')

            # If Multiclass classification case we calculate loss (categorical_crossentropy) and accuracy on train and val set
            if self.loss == 'categorical_cross_entropy':
                #Calculate train accuracy
                train_p = self.forward(x_train) 
                #Return max value/probability for each sample prediction and compare it with the true value
                i = np.argmax(train_p, axis=1) == np.argmax(y_train, axis=1) #Returns a bool array 
                train_acc = list(i).count(True) / len(i) * 100
                train_accs.append(train_acc)
                #Calculate validation accuracy
                if x_validation is not None:
                    val_p = self.forward(x_validation)
                    j = np.argmax(val_p, axis=1) == np.argmax(y_validation, axis=1)
                    val_acc = list(j).count(True)/len(j) * 100
                    val_accs.append(val_acc)
                print(f' - Loss: {np.round(loss,5)} \t Train Acc: {np.round(train_acc,2)} % \t  Val Acc: {np.round(val_acc,2)} %')
            
            #If Binary classification case
            if self.loss == 'binary_cross_entropy':
                #Calculate train accuracy
                train_p = self.forward(x_train)
                p1 = np.reshape(np.where(train_p > 0.5, 1, 0), -1) #Round predictions to integers (0 or 1)
                train_acc = np.sum((p1 == y_train)) / p1.shape[0] * 100 #Np.sum on bool array counts the True values
                train_accs.append(train_acc)
                #Calculate validation accuracy
                if x_validation is not None:
                    val_p = self.forward(x_validation)
                    p2 = np.reshape(np.where(val_p > 0.5, 1, 0), -1)
                    val_acc = np.sum((p2 == y_validation)) / p2.shape[0] * 100
                    val_accs.append(val_acc)
                print(f' - Loss: {np.round(loss,5)} \t Train Acc: {np.round(train_acc,2)} % \t  Val Acc: {np.round(val_acc,2)} %')

            #Early Stopping
            if self.earlystop is not None:
                #Save weights if restore_weights is true
                if self.earlystop.restore_weights:
                    for idx, layer in self.layers.items():
                        self.restore['W'+str(idx)+str(epoch)] = layer.weights
                        self.restore['b'+str(idx)+str(epoch)] = layer.bias
                    #Do nothing until the first 'patience' epochs are computed
                    if epoch >= self.earlystop.patience:
                        stop = self.earlystop.calculate_stop(val_accs, self.loss) #Calculate stop criterion
                        #Delete useless weights and bias
                        if epoch > self.earlystop.patience:
                            for idx, layer in self.layers.items():
                                del self.restore['W'+str(idx)+str(epoch - self.earlystop.patience)]
                                del self.restore['b'+str(idx)+str(epoch - self.earlystop.patience)]
                        if stop: # If stop criterion is reached
                            #Restore Weights
                            for idx, layer in self.layers.items():
                                layer.weights = self.restore['W'+str(idx)+str(epoch-self.earlystop.patience+1)]
                                layer.bias = self.restore['b'+str(idx)+str(epoch-self.earlystop.patience+1)]
                            break #Stop training
                #If restore_weights=False simply calculate when to stop
                else:
                    if epoch >= self.earlystop.patience:
                        stop = self.earlystop.calculate_stop(val_accs, self.loss) #Calculate stop criterion
                    if stop:
                        break

        self.history = {'loss': losses, 'train_acc': train_accs, 'val_acc': val_accs}


    def loss_function(self, y_true, y_pred, derivative=False):
        '''Contains the loss functions and their derivatives. Derivatives are used
            to calculate the gradient of the last layer while the functions are used
            to calculate the loss over each epoch.
            Implemented losses: mse, binary_cross_entropy, categorical_cross_entropy
            NB: categorical_cross_entropy requires one-hot encoded labels'''
        if self.loss == 'categorical_cross_entropy':
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) #Clip values to prevent overflow
            if derivative:
                return y_pred - y_true
            n = y_true.shape[0]
            loss = (-1.0) / n * np.sum(y_true * np.log(y_pred))
            return loss
        
        if self.loss == 'binary_cross_entropy':
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) #Clip values to prevent overflow
            if derivative:
                return y_pred - y_true
            loss =  (-1.0) * np.mean(np.dot(np.transpose(y_true), np.log(y_pred)) + np.dot(1 - np.transpose(y_true), np.log(1 - y_pred)))
            return loss

        if self.loss == 'mse':
            if derivative:
                return y_pred - y_true
            loss = np.mean(np.square(np.subtract(y_true,y_pred)))
            return loss
    
    def shuffle(x, y):
        '''Consistent method to shuffle  two lists/np.arrays of the same length using the same index list'''
        #Check if inputs are lists or np.arrays
        n_samples_x = x.shape[0] if hasattr(x, 'shape') else len(x)
        n_samples_y = y.shape[0] if hasattr(y, 'shape') else len(y)
        assert n_samples_x == n_samples_y #Checks whether features and labels have the same size
        indices = np.arange(x.shape[0]) #Create a list of indexes
        indices = np.random.permutation(indices) #Shuffle the indexes
        return x[indices], y[indices] #Return features and labels with indexes aligned

    @staticmethod
    def get_batches(x, y, batch_size):
        '''This method checks first whether the batch_size is a multiple of training set size
        and determines the number of batches to create. Then, shuffles the dataset and creates 
        batches as a list of tuples (batch_x, batch y)'''
        batches = []
        n_input = x.shape[0] 

        #If training samples are a multiple of batch_size 
        if n_input % batch_size == 0:
            n_batches = int(n_input / batch_size) #Round to lower integer
        else:
            n_batches = int(n_input / batch_size) + 1

        #Shuffle dataset
        x1, y1 = NeuralNetwork.shuffle(x, y)

        #Create batches
        for i in range(n_batches):
            batch_x = x1[batch_size * i : (i+1) * batch_size]
            batch_y = y1[batch_size * i : (i+1) * batch_size]
            batches.append((batch_x, batch_y))

        return batches