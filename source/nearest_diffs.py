# this file calculates the topK patches corresponding to each diff in the test file and stores in a csv.
# predicted patch is also stored in the csv.

import argparse
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm
from lmg_eval_modified import finding_bestK, finding_topK, clean_msg
import csv
from sys import exit


def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/lmg/test.pkl', help='the directory of our training data')

    parser.add_argument('-train_cc2ftr_data', type=str, default='./data/lmg/train_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-test_cc2ftr_data', type=str, default='./data/lmg/test_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-topK', type=int, default=10, help='the value of K')
    parser.add_argument('-name_of_csvfile', type=str, required=True, help='name of file in which the output is to be stored')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]

    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]

    train_ftr = pickle.load(open(params.train_cc2ftr_data, "rb"))   
    test_ftr = pickle.load(open(params.test_cc2ftr_data, "rb"))
    k = params.topK
    
    
    final_list=[['test_diff'],['given_LM'],['pred_diff'],['pred_LM'],['top1_diff'],['top1_LM'],['top2_diff'],['top2_LM']
     ,['top3_diff'],['top3_LM'],['top4_diff'],['top4_LM'],['top5_diff'],['top5_LM'],['top6_diff'],['top6_LM']
     ,['top7_diff'],['top7_LM'],['top8_diff'],['top8_LM'],['top9_diff'],['top9_LM'],['top10_diff'],['top10_LM']]
    
    if k>10:
        t = 11
        while t <= k:
            s = 'top'+str(t)+'_diff'
            s1 = 'top'+str(t)+'_LM'
            final_list.append([s])
            final_list.append([s1])
            t=t+1
    
    for i, (_) in enumerate(tqdm([i for i in range(test_ftr.shape[0])])):
        temp=[]
        element = test_ftr[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        dist_metric = cosine_similarity(X=train_ftr, Y=element)
        topK_index = finding_topK(dist_metric, topK=k)
        # taking top k diffs based on cosine similarity
        bestK = finding_bestK(diff_trains=train_diff, diff_test=test_diff[i], topK_index=topK_index)
        # bestK is the index of predicted log message
        predlm = train_msg[bestK].lower()
        givenlm = test_msg[i].lower()
        prediff = train_diff[bestK]
        final_list[0].append(test_diff[i])
        final_list[1].append(givenlm)
        final_list[2].append(prediff)
        final_list[3].append(predlm)
        x = 4
        for j in topK_index:
            final_list[x].append(train_diff[j])
            final_list[x+1].append(train_msg[j])
            x=x+2
    name_file = params.name_of_csvfile + '.csv'   
    with open(name_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(final_list)
