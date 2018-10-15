"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        h = np.maximum(X.dot(W1) + b1, 0) # using ReLU as activation
        scores = h.dot(W2) + b2
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        #for numerical stability
        #scores -= np.max(scores, axis=1, keepdims=True)

        f=np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)
        #calculate loss
        L=np.sum(-np.log(f)[np.arange(N), y])    
        loss = L/N
        # add regularization        
        loss += 0.5*(reg*np.sum(W1 * W1) + reg*np.sum(W2 * W2))
       
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        
        f[range(N),y] -= 1
        temp = np.ones(h.shape[0])        
        grads['W2'] = (f.T.dot(h)).T/N + reg*W2
        grads['b2'] = f.T.dot(temp)/N
        grads['W1'] = ((f.dot(W2.T)*(h>0)).T.dot(X)).T/N + reg*W1
        grads['b1'] = ((f.dot(W2.T)*(h>0)).T.dot(temp)).T/N
   
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            
            # same as in LinearClassifier
            batch_indices =np.random.choice(num_train, batch_size,  replace = True)   
            X_batch = X[batch_indices,:]
            y_batch = y[batch_indices]
            
            
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            
            # same as in LinearClassifier but for each parameter in self.params
            for p in self.params:
                self.params[p] = self.params[p] - grads[p] * learning_rate
                
            
            
            
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        #y = argmax(softmax(max(X*W1 + b1,0)*W2 + b2))
        
        h = np.maximum(X.dot(self.params['W1']) + self.params['b1'],0).dot(self.params['W2']) + self.params['b2']
        scores = np.exp(h)/np.sum(np.exp(h), axis=1, keepdims=True)
        y_pred = np.argmax(scores, axis = 1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred
    
    
    
    
    
    def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):

        best_net = None # store the best model into this 

        ############################################################################

        # TODO: Tune hyperparameters using the validation set. Store your best     #

        # trained model in best_net.                                               #

        #                                                                          #

        # To help debug your network, it may help to use visualizations similar to #

        # the  ones we used above; these visualizations will have significant      #

        # qualitative differences from the ones we saw above for the poorly tuned  #

        # network.                                                                 #

        #                                                                          #

        # Tweaking hyperparameters by hand can be fun, but you might find it useful#

        # to  write code to sweep through possible combinations of hyperparameters #

        # automatically like we did on the previous exercises.                     #

        ############################################################################
        results = {}
        best_val = -1
        all_classifiers = []
        input_size = 32 * 32 * 3
        hidden_size = 120
        num_classes = 10
        learning_rates = [4e-4,3e-4] 
        iterations = 1200
        
        learning_rate_decays = [0.95]
        regularization_strengths = [0,0.5,0.9]
        epochs =3 
        for epoch in range(epochs):
            for learning_rate in learning_rates:
                for learning_rate_decay in learning_rate_decays:   
                    for reg in regularization_strengths:
                        print('lr: ', learning_rate, ' lrd: ',learning_rate_decay, 'reg: ', reg) 
                        model = TwoLayerNet(input_size, hidden_size, num_classes)

                        model.train(X_train, y_train, X_val, y_val, learning_rate, learning_rate_decay, reg, iterations, batch_size=200, verbose=False)
                        all_classifiers.append(model)

                        train_predict = model.predict(X_train)
                        val_predict = model.predict(X_val)

                        train_acc = np.mean(y_train == train_predict)
                        val_acc = np.mean(y_val == val_predict)

                        results[(learning_rate, reg, learning_rate_decay, iterations)] = train_acc, val_acc   

                        if(best_val<val_acc):
                            best_val = val_acc
                            best_net = model

        # Print out results.
        for (learning_rate,reg, learning_rate_decay, iterations) in sorted(results):
            train_accuracy, val_accuracy = results[(learning_rate, reg, learning_rate_decay,iterations)]
            print('learning_rate %e reg %e learning_rate_decay %e iterations %d train accuracy: %f val accuracy: %f' % (
                  learning_rate, reg, learning_rate_decay,iterations, train_accuracy, val_accuracy))

        print('best validation accuracy achieved during validation: %f' % best_val)
        print('all_classifiers', all_classifiers)
        ############################################################################

        #                               END OF YOUR CODE                           #

        ############################################################################

        return best_net