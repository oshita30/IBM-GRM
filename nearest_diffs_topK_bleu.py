import argparse
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm
from lmg_eval import finding_bestK, finding_topK, clean_msg

from sys import exit
# this file calculates the top k training diffs based on bleu score for every test diff and stores in a csv file.

def finding_topK_bleu(diff_trains, diff_test, topK=1):
    diff_code_train = [d.lower().split() for d in diff_trains] #list of lists of tokenized code changes
    diff_code_test = diff_test.lower().split() #single tokenized test code change in question
    chencherry = SmoothingFunction()
    scores = [sentence_bleu(references=[diff_code_test], hypothesis=d, smoothing_function=chencherry.method1) for d in
            diff_code_train]
    
    scores = list(scores)
    topK_index_bleu = list()
    for i in range(topK):
        max_ = max(scores)
        index = scores.index(max_)
        topK_index_bleu.append(index)
        del scores[index]
    
    return topK_index_bleu


def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/lmg/test.pkl', help='the directory of our training data')
    parser.add_argument('-topK', type=int, default=10, help='value of k')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]
    k = params.topK
    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]

    
    
    final_list_bleu=[['test_diff'],['given_LM'],['pred_diff'],['pred_LM'],['top1_diff'],['top1_LM'],['top2_diff'],['top2_LM']
     ,['top3_diff'],['top3_LM'],['top4_diff'],['top4_LM'],['top5_diff'],['top5_LM'],['top6_diff'],['top6_LM']
     ,['top7_diff'],['top7_LM'],['top8_diff'],['top8_LM'],['top9_diff'],['top9_LM'],['top10_diff'],['top10_LM']]
    
    
    if k>10:
        t = 11
        while t <= k:
            s = 'top'+str(t)+'_diff'
            s1 = 'top'+str(t)+'_LM'
            final_list_bleu.append([s])
            final_list_bleu.append([s1])
            t=t+1
        
    
    for i, (_) in enumerate(tqdm([i for i in range(2216)])):
        temp=[]
        
        # not using topK based on cosine similarity
        bestK = finding_bestK(diff_trains=train_diff, diff_test=test_diff[i], topK_index=None)
        # bestK is the index of predicted log message
        predlm = train_msg[bestK].lower()
        givenlm = test_msg[i].lower()
        prediff = train_diff[bestK]
        final_list_bleu[0].append(test_diff[i])
        final_list_bleu[1].append(givenlm)
        final_list_bleu[2].append(prediff)
        final_list_bleu[3].append(predlm)
        x = 4
        # top k diffs based on bleu score
        topK_index_bleu = finding_topK_bleu(diff_trains=train_diff, diff_test=test_diff[i], topK=k)
        for j in topK_index_bleu:
            final_list_bleu[x].append(train_diff[j])
            final_list_bleu[x+1].append(train_msg[j])
            x=x+2
        
        
    with open('nearest_diff_topk_bleu.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(final_list_bleu) 
