import torch
import torch.nn as nn
import sys
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import read_data
from transformers import AdamW
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
import pandas as pd
import re
import numpy as np
import pandas as pd

from conver import conversion

base='Attributes and Sample Values/'

QUS={}
QUS['top wear'] = pd.read_csv(base+'Top Wear.csv')['Standardized attributes']
QUS['full wear']= pd.read_csv(base+'Full Wear.csv')['Standardized attributes']
QUS['foot wear']= pd.read_csv(base+'Foot Wear.csv')['Standardized attributes']
QUS['bottom wear']= pd.read_csv(base+'Bottom Wear.csv')['Standardized attributes']
QUS['accessories']= pd.read_csv(base+'Accessories.csv')['Standardized attributes']

def flip_validate():
    device = torch.device('cpu') # set the device to cpu
    if(torch.cuda.is_available()): # check if cuda is available
        device = torch.device('cuda:0') # if cuda, set device to cuda
    torch.cuda.empty_cache()

    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(device)
    model.load_state_dict(torch.load('./checkpoints/baseline.pth'))

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    data = pd.read_csv('Flipkart_complete_data/Complete Test Data/complete_data/test.csv')
    context_list = data["Words"]
    tag_list = data['Category']

    output_list = []
    processor = conversion()

    count = len(context_list)
    model.eval()

    for iteration in range(count):
        temp_output = []

        i=iteration
        context = processor.input(context_list[i])

        if i%100==0:
            print("Iteration ",i , "| ", count)

        for question in QUS[tag_list[i].lower()]:
            encoding = tokenizer(context, question, truncation=True, padding=True)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            input_ids = torch.as_tensor([input_ids]).to(device)
            attention_mask = torch.as_tensor([attention_mask]).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            start_scores, end_scores = outputs['start_logits'], outputs['end_logits']
            start_scores, end_scores = torch.nn.functional.softmax(start_scores[0,:]), torch.nn.functional.softmax(end_scores[0,:])
            start_index = torch.argmax(start_scores).item()
            end_index = torch.argmax(end_scores).item()

            start_score, end_score = start_scores[start_index].item(), end_scores[end_index].item()
            confidence = start_score*end_score

            try:
                if end_index>=start_index and end_index-start_index<5 and confidence>0.4:
                    start_index, end_index = encoding.token_to_chars(start_index).start, encoding.token_to_chars(end_index).end
                    if start_index>=end_index:
                        continue
                    processor.assign(question,start_index,end_index)
                    answer = processor.output

                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                answer = tokenizer.convert_tokens_to_string(all_tokens[start_index: end_index + 1])

            except:
                continue

        output_list.append(processor.output)

    data["Tags"]  = output_list
    data.to_csv("results_complete.csv")

flip_validate()
