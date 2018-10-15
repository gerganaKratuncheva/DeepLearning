from random import shuffle
import numpy as np
import torch.utils.data.sampler as sampler
import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could look something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        # won't be using cuda for now 
        # TODO: check if cuda can be integrated
        num_iterations = num_epochs * iter_per_epoch
        for epoch in range(num_epochs):
            #for iteration in range(iter_per_epoch):  

            loss = 0.0
            val_loss = 0.0
            for i, data in enumerate(train_loader): 
                inputs, targets = data
                inputs = Variable(inputs) 
                targets = Variable(targets.type(torch.LongTensor ))
                # zero the parameter gradients
                optim.zero_grad()
                #forward pass
                out = model.forward(inputs)
                # loss
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(out, targets)
                # store loss 
                self.train_loss_history.append(loss)
                
                #log loss every log_nth iteration
                if (i+epoch*iter_per_epoch)%log_nth==0:
                    print('[Iteration %d/%d] Train loss: %.3f' %(i+epoch*iter_per_epoch, num_iterations, loss))
                    
                #backward pass
                loss.backward()
                #optimize
                optim.step()
            #After one epoch the training accuracy of the last mini batch is logged and stored in self.train_acc_history.
            train_accuracy = int((torch.max(out, 1)[1] == targets).sum())/out.size(0)
            self.train_acc_history.append(train_accuracy)
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' %(epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1]))
            #validate at the end of each epoch, log the result and store the accuracy of the entire validation set in self.val_acc_history
            val_correct = 0
            val_total = 0
            
            for i, data in enumerate(val_loader): 
                val_inputs, val_targets = data
                val_inputs = Variable(val_inputs) 
                val_targets = Variable(val_targets.type(torch.LongTensor ))
                val_out = model.forward(val_inputs)
                loss_func_val = torch.nn.CrossEntropyLoss()
                val_loss = loss_func(val_out, val_targets)

                val_correct += int((torch.max(val_out,1)[1] == val_targets).sum())
                val_total += val_out.size(0)
            
            val_accuracy = val_correct/val_total
            self.val_acc_history.append(val_accuracy)
            print('[Epoch %d/%d] VAL acc/loss: %.3f/%.3f' %(epoch, num_epochs, self.val_acc_history[-1], val_loss))

        
 
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
