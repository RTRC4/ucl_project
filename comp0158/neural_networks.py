

import tensorflow as tf



class network_varied_parameters():
    """Class object to build neural network with random parameter selection"""
    
    def __init__(self,lr, epsilon, num_neurons,num_layers,input_shape,epochs):
        self.opt=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)
        self.loss=tf.keras.losses.MeanAbsoluteError()
        self.num_neurons=num_neurons
        self.num_layers=num_layers
        self.input_shape=input_shape
        self.epochs=epochs                   
        
        
        
    def construct_model(self):
        """Function to construct the tensorflow neural network model"""
            
        model=tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.num_neurons,activation='relu', input_shape=(self.input_shape,)))
        if self.num_layers==2:
            model.add(tf.keras.layers.Dense(self.num_neurons,activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        return model
    
        

    def train_model(self,train_X,train_y,val_X,val_y,model=None):
        """Function to train a neural network model"""
        
        val_loss_lt=list()
        if model is None: model=self.construct_model()
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                y=model(train_X, training=True)
                train_loss=self.loss(train_y,y)
            grads=tape.gradient(train_loss, model.trainable_variables)
            self.opt.apply_gradients(zip(grads, model.trainable_variables))    
            
            val_pred=model(val_X)
            val_loss=self.loss(val_y,val_pred)
            val_loss_lt.append(val_loss)
            if val_loss > val_loss_lt[i-1]:
                break
            
        return val_pred, model, val_loss
        


