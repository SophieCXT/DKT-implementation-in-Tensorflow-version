# coding: utf-8

# to get information from the csv, and utilize generator
def getInfo(csvInput):
    
    
    # in csv, every 3 row is a unit including studentID, questionSet, and resultSet(correct = 1 wrong = 0)  
    for row in csvInput:
        # nStep, questionsID, correct = yield(line)
        if row[-1]=='':
            yield row[:-1] # to generate questionSet and resultSet without ''
        else: yield row # to generate studentID
