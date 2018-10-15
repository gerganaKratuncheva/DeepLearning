"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    N = X.shape[0]
    C = W.shape[1]
    s_i_full = np.zeros(shape = (N,C))
    x_temp = np.zeros(shape = (1,X.shape[1]))  
    for i in range(N):
        #s_i = sum_{j=1} X_j * W_ji              
        for b in range(C):
            x_temp = X[i]
            for a in range(x_temp.shape[0]):           
                s_i_full[i][b] += x_temp[a]*W[a][b]
            #to avoid numeric instability (risk of blowup), normalize by shifting the values s.t. the maximal value is 0
            s_i_full[i] -= np.max(s_i_full[i])
        su = 0.0
        for k in range(len(s_i_full[i])):
            su += np.exp(s_i_full[i][k]) 

        loss += -np.log(np.exp(s_i_full[i][y[i]]) / su)
        for c in range(C):
            f = np.exp(s_i_full[i][c])/su                      
            np.add(dW[:, c], (f-(c == y[i])) * X[i], out=dW[:, c], casting="unsafe")

    loss /= N    
    # add regularization to the loss: R = (||W||_2)^2
    R = 0.0
    for n in range(W.shape[0]):
        for m in range(W.shape[1]):
            R += (W[n][m] ** 2)
    # the whole regularization term is: lambda * R / 2; lambda=reg            
    loss += reg * R/2
    
    dW = dW/N
    dW += reg*W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    N = X.shape[0]

    # calculate softmax and
    # normalizee to avoid numeric instability
    s = X.dot(W)
    s -= np.max(s, axis=1, keepdims=True)
    #calculate softmax
    f=np.exp(s)/np.sum(np.exp(s), axis=1, keepdims=True)
    #calculate loss
    L=np.sum(-np.log(f)[np.arange(N), y])    
    # add regularization
    loss =L/N + 0.5 * reg * np.linalg.norm(W)**2

    
    # use softmax matrix f and find correct
    f[np.arange(N),y] -= 1
    dW = X.T.dot(f/N) 
    # add regularization
    dW += reg*W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7, 4.5e-7]
    regularization_strengths = [1.5e3, 2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    #as shown in the lecture:
    for lr in learning_rates:
        for reg_str in regularization_strengths:
            # classifier
            model = SoftmaxClassifier()
            # train a classifier on the training set for each learning_rate = lr and regularization_strength = reg_str
            model.train( X_train, y_train, lr, reg_str, num_iters=2000, batch_size = 200, verbose=False)
            # add classifier to all_classifiers            
            all_classifiers.append(model)
            # compute predictions to use to compute accuracy
            train_predict = model.predict(X_train)
            val_predict = model.predict(X_val)
            
            # compute accuracy 
            train_acc = np.mean(y_train == train_predict)
            val_acc = np.mean(y_val == val_predict)
            print('reg strength: ', reg_str, 'learning rate:',lr,'val accuracy:',val_acc)
            #  store accuracies in the results dictionary
            results[(lr, reg_str)] = train_acc, val_acc   
    
            # store the best validation accuracy in best_val and the Softmax object that achieves this accuracy in best_softmax
            if(best_val<val_acc):
                best_val = val_acc
                best_softmax = model
            
        
        
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)
    print('all_classifiers', all_classifiers)
    return best_softmax, results, all_classifiers
