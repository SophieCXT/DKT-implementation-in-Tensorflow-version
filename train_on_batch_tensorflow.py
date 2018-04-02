import tensorflow as tf
import models
import data_reader

epoch=10
batch_size=16

dataReader=data_reader.DataReader()
input_dim_order =dataReader.get_input_dim_order()
input_dim = 2 * input_dim_order
dataLongest=100
print(input_dim_order)
epoch = 10
hidden_layer_size = 512
keep_prob=1.0 #here the default keep_prob is 1.0
net=models.DKTnet(input_dim,input_dim_order,hidden_layer_size,keep_prob,dataLongest-1)

with tf.Session() as sess:
    #training
    sess.run(net.init)
    for i in range(0,epoch):
        for j in range(0,int(dataReader.get_data_count()/batch_size)):
            batch_x,batch_y,batch_y_order=dataReader.next_train_batch(batch_size)
            loss,_=sess.run((net.loss,net.train_op),feed_dict=net.dict(batch_x,batch_y,batch_y_order))
            if((j+1)%10==0):
                print("The %d epoch %d batch the loss is: %.4f"%(i,j,loss))

        #validating
        val_x,val_y,val_y_order=dataReader.validation_set()
        tp,tn,fp,fn,acc=sess.run((net.tp,net.tn,net.fp,net.fn,net.acc),feed_dict=net.dict(val_x,val_y,val_y_order))
        print("For the epoch %d the val acc is: %.4f"%(i,acc))
