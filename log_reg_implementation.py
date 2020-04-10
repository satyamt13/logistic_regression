import numpy as np

class log_reg():
    
    def __init__(self):
        self._learning_rate = None
        self._X = None
        self._y = None
        self._theta = None
        
    def sigmoid(self,t):
        return 1/( 1 + np.exp(-t) )
    
    def h(self,X):
        return self.sigmoid( np.dot( X, self._theta.T ) )
    
    def compute_loss(self):
        return (1/self._X.shape[0]) * np.sum( - self._y * np.log( self.h(self._X) ) - ( 1 - self._y ) * np.log( 1 - self.h(self._X) ) )
    
    def train(self, X, y, epochs = 100, learning_rate = 0.1):
        self._X = X = np.hstack( ( X, np.ones((X.shape[0],1)) ) )
        self._y = y
        self._theta = np.zeros( ( 1, self._X.shape[1] ) )
        current_epoch = 0
        while current_epoch < epochs:
            for i in range(self._X.shape[0]):
                h_x = self.h( self._X[i] )
                gradient = ( h_x - self._y[i].reshape(1,-1) ) * self._X[i].reshape(1,-1)
                self._theta -= learning_rate * gradient 
            print( "Epoch: {}, Loss: {}".format( current_epoch,self.compute_loss() ) )
            current_epoch += 1 

