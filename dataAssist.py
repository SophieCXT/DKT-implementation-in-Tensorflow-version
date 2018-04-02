# coding: utf-8
import csv
import numpy as np
import utils

max_train = None
max_steps = None

class DataAssistMatrix():
    
    def __init__(self):
        
        print('Build a DataAssistMatrix')

        self.longest = 100
        self.questions = {} # use dictionary to remark questionID
        self.n_questions = 0 # total number of distinct quetions
        self.max_questionID = 0
        self.trainData = []

    def build(self):
        print('Loading data...')
        #training process
        root = './'
        trainPath = root + 'data/assistments/builder_train.csv'
        csvFile = open(trainPath, 'r')
        csvInput = csv.reader(csvFile)
        trainData = []

        totalAnswers = 0
        student_num = 0
        
        while(True):
            student = self.loadInfo(csvInput)
            if student == None:
                break
            student_num += 1
            if(student.n_answers >= 2 and student.n_answers<=self.longest):
                trainData.append(student)
            elif student.n_answers > self.longest:
                student.n_answers = self.longest
                student.ID = student.ID[:self.longest]
                student.correct = student.correct[:self.longest]
                trainData.append(student)
                
            if len(trainData) % 1000 == 0:
                print ('The length of train data is now ',len(trainData))
            totalAnswers = totalAnswers + student.n_answers
        self.trainData = trainData
        self.max_questionID = self.n_questions # because of dence ID
        print ('The num of all students is ', student_num)
        csvFile.close()


    def loadInfo(self, csvInput):
        
        getInfo = utils.getInfo(csvInput)
        
        try:
            studentID = next(getInfo) 
            questionsID = next(getInfo)
            correct = next(getInfo)
        except:
            # print ('Execption occurs in function--loadInfo')
            return None

#       print ('-------------loadInfo begin-----------')
        n = int(studentID[0])
#       print ('studentID[0] is',n)
#       print ('-------------------------------------')
        
        newID = [] # to replace questionSet by simplifying each questions serial number

        # use replacement to shorten one hot list length
        for i in range(len(questionsID)):
            if not questionsID[i] in self.questions:
                self.questions.update({questionsID[i]:self.n_questions})
                self.n_questions += 1
            newID.append(self.questions[questionsID[i]])
            
#        print ('questionSet is ',questionSet)
#        print ('-------------------------------------')
#        print ('newQuestionSet is',newQuestionSet)
#        print ('-------------------------------------')

        stu = student(n, newID, correct)
        return stu

class student():
    def __init__(self,n,ID,correct):
        self.n_answers = n
        self.ID = np.zeros(n,int)
        self.correct = np.zeros(n,int)



