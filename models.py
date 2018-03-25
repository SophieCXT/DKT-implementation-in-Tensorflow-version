import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim


def tlength(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence+1), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


class DKTnet:

    def dict(self,X,Y,Order):
             return {self._X:X,self._Y:Y,self._Order:Order}


    def __init__(self,input_dim,input_dim_order,hidden_size,keep_prob,longest):
         self._X=tf.placeholder(tf.float32,[None,longest,input_dim])
         self._Order=tf.placeholder(tf.float32,[None,longest,input_dim_order])
         self._Y=tf.placeholder(tf.float32,[None,longest])

         #lstm layer
         lstm_cell=rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
         lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
         output, state = tf.nn.dynamic_rnn(lstm_cell,
         self._X,
         dtype=tf.float32,
         sequence_length=tlength(self._X),
         )

         #fully connection
         outputDense=tf.reshape(output,[-1,hidden_size])
         dense=slim.fully_connected(outputDense,input_dim_order)
         #activitaion
         dense=tf.nn.sigmoid(dense) 

         #All the scores
         scores=tf.reshape(dense,[-1,longest,input_dim_order])

         #certain question scores
         result=tf.reduce_max(scores*self._Order,2)

         #mask the useless output
         mask = tf.sequence_mask(tlength(self._X), maxlen=longest, dtype=tf.float32)
         E=1e-8
         self.loss=-(self._Y*tf.log(result+E)*mask+(1-self._Y)*(tf.log(1-result-E))*mask)

         #loss function
         self.loss=tf.reduce_sum(self.loss) / tf.reduce_sum(mask)

         #optimizer
         self.train_op=tf.train.AdamOptimizer(0.001).minimize(self.loss)

         #calculate the tp tn fp fn, The threshold is 0.5
         self.tp=tf.reduce_sum(mask*tf.to_float(tf.greater(result*self._Y,0.5)))
         self.fp=tf.reduce_sum(mask*tf.to_float(tf.greater((1-result)*self._Y,0.5)))
         self.tn=tf.reduce_sum(mask*tf.to_float(tf.greater(result*(1-self._Y),0.5)))
         self.fn=tf.reduce_sum(mask*tf.to_float(tf.greater((1-result)*(1-self._Y),0.5)))

         #calculate the accuracy
         self.acc=(self.tp+self.fp+E)/(self.tp+self.tn+self.fp+self.fn+E)

         #variable initializer
         self.init=tf.global_variables_initializer()
