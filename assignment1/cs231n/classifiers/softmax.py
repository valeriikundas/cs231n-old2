import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    sum_up = np.sum(X[i] * W.T[y[i]])
    up = np.exp(sum_up)
    sum_down = np.sum(X[i].dot(W))
    down = np.exp(sum_down)
    loss += -np.log(up / down)
    
    dfraction = -1 / (up / down)
    dup = dfraction * (1 / down)
    dinv_down = dfraction * up
    ddown = dinv_down * (-1 / (down ** 2))
    
    dsum_up = np.exp(sum_up) * dup
    dmul_up = np.full((1, X.shape[1]), dsum_up)
    dsum_down = np.exp(sum_down) * ddown
    dmul_down = np.full((1, num_classes), dsum_down)
    
    dW += X[i].T[:, np.newaxis].dot(dmul_down)
    
    dXWy = dmul_up * X[i]
    dW[:, y[i]][:, np.newaxis] += dXWy.T
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W * W)  
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  sum_scores = np.sum(scores, axis=1)
  exp_sum_scores = np.exp(sum_scores) # down
  inv_exp_scores = 1 / exp_sum_scores
  
  cor_W = np.zeros(X.shape)
  for i in range(X.shape[0]):
    cor_W[i] = W.T[y[i]]
  cor_scores = np.sum(X * cor_W, axis=1)
  exp_cor_scores = np.exp(cor_scores) # up
  
  fraction = exp_cor_scores * inv_exp_scores
  loss_all = -np.log(fraction)
  loss = np.sum(loss_all) / X.shape[0]
    
  loss += 2 * reg * np.sum(W * W)

  dloss_all = np.ones((X.shape[0], 1))
  dfraction = dloss_all * (-1 / (dloss_all ** 2))

  dinv_exp_sum_scores = dfraction * exp_cor_scores[:, np.newaxis]

  dexp_sum_scores = dinv_exp_sum_scores.reshape(-1) * (-1 / (exp_sum_scores ** 2)) 
  dsum_scores = dexp_sum_scores * np.exp(sum_scores)
  dscores = np.tile(dsum_scores.reshape(1, -1), (W.shape[1], 1))
  dW += X.T.dot(dscores.T)

  dexp_cor_scores = dfraction.reshape(-1) * dexp_sum_scores
  dcor_sum_scores = dexp_cor_scores * np.exp(cor_scores)
  dcor_scores = np.tile(dcor_sum_scores, (X.shape[1], 1))
  
  dcor_W = dcor_scores.T * X
  
  for i in range(X.shape[0]):
    dW[:, y[i]] += dcor_W[i].T

  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

