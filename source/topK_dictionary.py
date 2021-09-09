# stores the BLEU score for different values of K in a dictionary and dumps the pkl file into a target folder
# graph can be plotted using this dictionary
import argparse
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm
from lmg_eval import load_kNN_model, clean_msg, finding_topK, finding_bestK, get_data_index, clean_each_line
import matplotlib.pyplot as plt
from sys import exit
from pickle import dump


def read_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_data', type=str, default='./data/lmg/train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/lmg/test.pkl', help='the directory of our training data')

    parser.add_argument('-train_cc2ftr_data', type=str, default='./data/lmg/train_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-test_cc2ftr_data', type=str, default='./data/lmg/test_cc2ftr.pkl', help='the directory of our training data')
    parser.add_argument('-lower_limit', type=int, default='2', help='lower value of K')
    parser.add_argument('-upper_limit', type=int, default='20', help='upper value of K')
    parser.add_argument('-step', type=int, default='1', help='step size between lower and upper limit')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    data_train = pickle.load(open(params.train_data, "rb"))
    train_msg, train_diff = clean_msg(data_train[0]), data_train[1]

    data_test = pickle.load(open(params.test_data, "rb"))
    test_msg, test_diff = data_test[0], data_test[1]

    train_ftr = pickle.load(open(params.train_cc2ftr_data, "rb"))   
    test_ftr = pickle.load(open(params.test_cc2ftr_data, "rb"))
    

    org_diff_data = (train_diff, test_diff)
    tf_diff_data = (train_ftr, test_ftr)
    ref_data = (train_msg, test_msg)
    
    dict_k={}
    x1 = params.lower_limit
    x2 = params.upper_limit + 1
    h = params.step
    for k in range(x1,x2,h):
        blue_scores = load_kNN_model(org_diff_code=org_diff_data, tf_diff_code=tf_diff_data, ref_msg=ref_data, topK=k)
        dict_k[k] = sum(blue_scores) / len(blue_scores) * 100
        print('Average of blue scores for k=',k,': ', sum(blue_scores) / len(blue_scores) * 100)
        
    a_file = open("dict_k.pkl", "wb")
    pickle.dump(dict_k, a_file)
    a_file.close()
