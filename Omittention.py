import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

max_len = 200
rnn_cell_size = 128
vocab_size=250

class StackAttention(tf.keras.layers.Layer): #https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(StackAttention, self).__init__(**kwargs)
        

    
    #You need to override the __init__ function. See the link below for details:
    #https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf/58799021        
    
    def get_config(self): 
        config = super(StackAttention, self).get_config().copy()
        config.update({'return_sequences' : self.return_sequences})
        return config        

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(StackAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

class KTAttention(tf.keras.Model): #https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
    def __init__(self, units):
        super(KTAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class SuperAttention(tf.keras.layers.Layer): #https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
    def __init__(self,**kwargs):
        super(SuperAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(SuperAttention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(SuperAttention,self).get_config()

