# same as kNN_cosine_jaccard, just jaccard replaced by edit_distance
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
def get_data_index(data, indexes):
    return [data[i] for i in indexes]

def clean_msg(messages):
    return [clean_each_line(line=msg) for msg in messages]

def clean_each_line(line):
    line = line.strip()
    line = line.split()
    line = ' '.join(line).strip()
    return line

def editDistance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    DP = [[0 for i in range(len1 + 1)]
             for j in range(2)];
    
    for i in range(0, len1 + 1):
        DP[0][i] = i

    for i in range(1, len2 + 1):
        for j in range(0, len1 + 1):
            if (j == 0):
                DP[i % 2][j] = i

            elif(str1[j - 1] == str2[i-1]):
                DP[i % 2][j] = DP[(i - 1) % 2][j - 1]

            else:
                DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j],
                                    min(DP[i % 2][j - 1],
                                  DP[(i - 1) % 2][j - 1])))

    return DP[len2 % 2][len1]

def finding_topK_editDist(diff_trains, diff_test, topK=1):
    scores = [editDistance(d, diff_test) for d in diff_trains]
    
    scores = list(scores)
    topK_index = list()
    for i in range(topK):
        min_ = min(scores)
        index = scores.index(min_)
        topK_index.append(index)
        del scores[index]
    
    return topK_index









def finding_topK_jaccard(diff_trains, diff_test, topK=1):
    diff_code_train = [d.lower().split() for d in diff_trains] #list of lists of tokenized code changes
    diff_code_test = diff_test.lower().split() #single tokenized test code change 
    
    for i in range(len(diff_code_train)):
        j = 0
        while j<len(diff_code_train[i]):
            if diff_code_train[i][j].isalnum():
                j=j+1
                continue
            else:
                del diff_code_train[i][j]
    
    i=0
    while i<len(diff_code_test):
        if diff_code_test[i].isalnum():
            i=i+1
            continue
        else:
            del diff_code_test[i]
    
    diff_test=set()
    for w in diff_code_test:
        diff_test.add(w)
    
    diff_train=[]
    for y in diff_code_train:
        temp=set()
        for w in y:
            temp.add(w)
        diff_train.append(temp)
    
    scores = [len(d.intersection(diff_test))/len(d.union(diff_test)) for d in diff_train]
    
    scores = list(scores)
    topK_index_jaccard = list()
    for i in range(topK):
        max_ = max(scores)
        index = scores.index(max_)
        topK_index_jaccard.append(index)
        del scores[index]
    
    return topK_index_jaccard

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
    parser.add_argument('-topK_jaccard', type=int, default=1, help='value of k for jaccard similarity')
    parser.add_argument('-topK_cosine', type=int, default=20, help='value of k for cosine similarity')
    parser.add_argument('-train_cc2ftr_data', type=str, default='./data/lmg/train_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-test_cc2ftr_data', type=str, default='./data/lmg/test_cc2ftr.pkl', help='the directory of our training data')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]
    k_jac = params.topK_jaccard
    k_cos = params.topK_cosine
    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]
    train_ftr = pickle.load(open(params.train_cc2ftr_data, "rb"))   
    test_ftr = pickle.load(open(params.test_cc2ftr_data, "rb"))
    
    final_list_bleu=[['test_diff'],['given_LM'],['pred_diff'],['pred_LM'],['bleu_score_pred_given']]
    
    t=1
    while t <= k_cos:
        s = 'top'+str(t)+'_diff'
        s1 = 'top'+str(t)+'_LM'
        final_list_bleu.append([s])
        final_list_bleu.append([s1])
        t=t+1
        
    bleu_scores=[]
    for i, (_) in enumerate(tqdm([i for i in range(2216)])):
        # now i have to find topK_cos from train_diff_new
        element = test_ftr[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        cosine_sim = cosine_similarity(X=train_ftr, Y=element)
        topK_index_cos = finding_topK(cosine_sim=cosine_sim, topK=k_cos)
        train_diff_new = [train_diff[x] for x in topK_index_cos]
        train_msg_new = [train_msg[w] for w in topK_index_cos]
        train_ftr_new = train_ftr[topK_index_cos]
        topK_index_editD = finding_topK_editDist(diff_trains=train_diff_new, diff_test=test_diff[i], topK=1)           
        bestK = topK_index_editD[0]
        # bestK is the index of predicted log message
        predlm = train_msg_new[bestK].lower()
        givenlm = test_msg[i].lower()
        prediff = train_diff_new[bestK]
        final_list_bleu[0].append(test_diff[i].replace('<nl>','\n'))
        final_list_bleu[1].append(givenlm)
        final_list_bleu[2].append(prediff.replace('<nl>','\n'))
        final_list_bleu[3].append(predlm)
        chencherry = SmoothingFunction()
        blue_score = sentence_bleu(references=[givenlm.split()], hypothesis=predlm.split(),smoothing_function=chencherry.method5)
        final_list_bleu[4].append(blue_score)
        bleu_scores.append(blue_score)
        x = 5
        
        for j in topK_index_cos:
            final_list_bleu[x].append(train_diff[j])
            final_list_bleu[x+1].append(train_msg[j])
            x=x+2
        
    print('Average of blue scores for k=',k_cos,' :', sum(bleu_scores) / len(bleu_scores) * 100)
    print('size of test data = ', len(bleu_scores))
