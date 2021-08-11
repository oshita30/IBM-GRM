#requirements:
#!pip install bert-extractive-summarizer==0.4.2
#!pip install sentencepiece
#!pip install transformers==3.3.0

from summarizer import Summarizer
import argparse
import csv
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm
from sys import exit
import gc
# this file calculates the top k training diffs based on bleu score for every test diff and stores in a csv file.
def get_data_index(data, indexes):
    return [data[i] for i in indexes]

def clean_msg(messages):
    return [clean_each_line(line=msg) for msg in messages]

def clean_each_line(line):
    line = line.strip()
    line = line.split()
    line = ' '.join(line).strip()
    return line

def finding_topK(cosine_sim, topK):
    cosine_sim = list(cosine_sim)
    topK_index = list()
    for i in range(topK):
        max_ = max(cosine_sim)
        index = cosine_sim.index(max_)
        topK_index.append(index)
        del cosine_sim[index]
    return topK_index

def finding_bestK(diff_trains, diff_test, topK_index):
    if topK_index == None:        
        diff_code_train = [d.lower().split() for d in diff_trains]
    else:
        diff_code_train = get_data_index(data=diff_trains, indexes=topK_index)
        diff_code_train = [d.lower().split() for d in diff_code_train]

    diff_code_test = diff_test.lower().split()
    chencherry = SmoothingFunction()
    scores = [sentence_bleu(references=[diff_code_test], hypothesis=d, smoothing_function=chencherry.method1) for d in diff_code_train]
    bestK = scores.index(max(scores))
    
    return bestK
                    
                    
def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/lmg/test.pkl', help='the directory of our training data')
    parser.add_argument('-topK_cosine', type=int, default=100, help='value of k for cosine similarity')
    parser.add_argument('-train_cc2ftr_data', type=str, default='./data/lmg/train_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-test_cc2ftr_data', type=str, default='./data/lmg/test_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-csv_name', type=str, default='kNN_cosine_BERT_.csv', help='name of csv file')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]
    k_cos = params.topK_cosine
    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]
    train_ftr = pickle.load(open(params.train_cc2ftr_data, "rb"))   
    test_ftr = pickle.load(open(params.test_cc2ftr_data, "rb"))
    name = params.csv_name
    list1=[['test_diff','given msg','pred msg','bleu']]        
    bleu_scores=[]
    for i, (_) in enumerate(tqdm([i for i in range(2216)])):
        # now i have to find topK_cos from train_diff_new
        element = test_ftr[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        cosine_sim = cosine_similarity(X=train_ftr, Y=element)
        topK_index_cos = finding_topK(cosine_sim=cosine_sim, topK=k_cos)
        train_diff_new = [train_diff[x] for x in topK_index_cos]
        
        sum_text=''
        for x in topK_index_cos:
          tmp = train_msg[x].replace('.',' ')
          sum_text = sum_text + tmp + '.'
          
        train_ftr_new = train_ftr[topK_index_cos]
        model=Summarizer()
        predlm = model(sum_text,num_sentences=0).lower()
        givenlm = test_msg[i].lower()
        givenlm = givenlm.replace(',',' ')
        chencherry = SmoothingFunction()
        blue_score = sentence_bleu(references=[givenlm.split()], hypothesis=predlm.split(),smoothing_function=chencherry.method5)
        bleu_scores.append(blue_score)
        list1.append([test_diff[i],givenlm,predlm,blue_score])
        gc.collect()
        with open(name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(list1)
    print('Average of blue scores for k= ',k_cos,': ', sum(bleu_scores) / len(bleu_scores) * 100)
    print('size of test data = ', len(bleu_scores))    

