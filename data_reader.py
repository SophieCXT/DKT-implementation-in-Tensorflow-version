import numpy as np
import csv
import utils
from dataAssist import DataAssistMatrix
import numpy as np
import utils
import pdb
from models import *
import pdb
import pickle
from dataAssist import DataAssistMatrix, student
import random
import sys

class DataReader:
    def __init__(self):
        data = DataAssistMatrix()
        data.build()
        self.input_dim_order =  int(data.max_questionID + 1)
        self.input_dim = 2 * self.input_dim_order
        input_dim_order=int(data.max_questionID + 1)
        input_dim=2 * self.input_dim_order
        epoch = 10
        hidden_layer_size = 512
        self.validation_split = 0.2
        train_data = []
        self.batch_index=0

        for student in data.trainData:
            train_data.append(student)

        print('The total size of raw data is: ', sys.getsizeof(data.trainData))

        x_train = []
        y_train = []
        y_train_order = []
        num_student = 0 # num of TRAINING student in each epoch

        '''Training part starts from now'''
        random.shuffle(train_data)
        print('Training data is shuffled')
        for student in train_data:
            num_student += 1
            x_single_train = np.zeros([input_dim, data.longest])
            y_single_train = np.zeros([1, data.longest])
            y_single_train_order = np.zeros([input_dim_order, data.longest])

            for i in range(student.n_answers):
                if student.correct[i] == 1.: # if correct
                    x_single_train[student.ID[i]*2-1, i] = 1.
                elif student.correct[i] == 0.: # if wrong
                    x_single_train[student.ID[i]*2, i] = 1.
                else:
                    print (student.correct[i])
                    print ("wrong length with student's n_answers or correct")
                y_single_train[0, i] = student.correct[i]
                y_single_train_order[student.ID[i], i] = 1.

            for i in  range(data.longest-student.n_answers):
                x_single_train[:,student.n_answers + i] = -1
                y_single_train[:,student.n_answers + i] = 0
                #notice that the padding value of order is still zero.
                y_single_train_order[:,student.n_answers + i] = 0
            x_single_train = np.transpose(x_single_train)
            y_single_train = np.transpose(y_single_train)
            y_single_train_order = np.transpose(y_single_train_order)
            x_train.append(x_single_train)
            y_train.append(y_single_train.reshape((data.longest)))
            y_train_order.append(y_single_train_order)
            
            
            self.x_train=np.array(x_train)
            self.x_train=self.x_train[:,:-1,:]
            self.y_train=np.array(y_train)
            self.y_train=self.y_train[:,1:]
            self.y_train_order=np.array(y_train_order)
            self.y_train_order=self.y_train_order[:,1:,:]

    def next_train_batch(self,batch_size):
        end=int(len(self.x_train)*self.validation_split)
        if(self.batch_index+batch_size>end):
            begin=self.batch_index
            self.batch_index=0
            return self.x_train[begin:end],self.y_train[begin:end],self.y_train_order[begin:end]
        else:
            begin=self.batch_index
            self.batch_index=self.batch_index+batch_size
            return self.x_train[begin:self.batch_index],self.y_train[begin:self.batch_index],self.y_train_order[begin:self.batch_index]

    def validation_set(self):
        begin=int(len(self.x_train)*self.validation_split)
        end=len(self.x_train)
        return self.x_train[begin:end],self.y_train[begin:end],self.y_train_order[begin:end]

    def get_input_dim_order(self):
        return self.input_dim_order

    def get_data_count(self):
        return len(self.x_train)
