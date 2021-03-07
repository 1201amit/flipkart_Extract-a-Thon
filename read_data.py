import glob, os
import pandas as pd 
import numpy as np 
import sys 
import json 

def readSupervisedData(data_path):
    with open(data_path) as f:
        data = json.load(f)
    
    context = []
    question = []
    answer = []
    start = []
    end = []
    for i in range(len(data)):
        context.append(data[i]['Context'])
        question.append(data[i]['Qus'])
        answer.append(data[i]['Ans'])
        start.append(data[i]['start'])
        end.append(data[i]['end'])

    return context, question, answer, start, end 

