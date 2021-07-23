import argparse
import pickle
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import re
from tqdm import tqdm
from sys import exit
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
    parser.add_argument('-topK_jaccard', type=int, default=100, help='value of k for jaccard similarity')
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
    
    final_list_bleu=[['test_diff'],['given_LM'],['pred_diff'],['pred_LM'],['bleu_score_pred_given'],['top1_diff'],['top1_LM'],['top2_diff'],['top2_LM']
     ,['top3_diff'],['top3_LM'],['top4_diff'],['top4_LM'],['top5_diff'],['top5_LM'],['top6_diff'],['top6_LM']
     ,['top7_diff'],['top7_LM'],['top8_diff'],['top8_LM'],['top9_diff'],['top9_LM'],['top10_diff'],['top10_LM']]
    
    
    if k_cos>10:
        t = 11
        while t <= k_cos:
            s = 'top'+str(t)+'_diff'
            s1 = 'top'+str(t)+'_LM'
            final_list_bleu.append([s])
            final_list_bleu.append([s1])
            t=t+1
        
    bleu_scores=[]
    for i, (_) in enumerate(tqdm([i for i in range(2216)])):
       
        topK_index_jaccard = finding_topK_jaccard(diff_trains=train_diff, diff_test=test_diff[i], topK=k_jac)
        train_diff_new = [train_diff[x] for x in topK_index_jaccard]
        train_msg_new = [train_msg[w] for w in topK_index_jaccard]
        topK_index_jaccard = np.array(topK_index_jaccard)
        # finding topK_jac code changes based on jaccard similarity
        # now my training set is reduced to topK_jac code changes
        train_ftr_new = train_ftr[topK_index_jaccard]
        
        # now i have to find topK_cos from train_diff_new
        element = test_ftr[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        cosine_sim = cosine_similarity(X=train_ftr_new, Y=element)
        topK_index_cos = finding_topK(cosine_sim=cosine_sim, topK=k_cos)
                    
        bestK = finding_bestK(diff_trains=train_diff_new, diff_test=test_diff[i], topK_index=topK_index_cos)
        # bestK is the index of predicted log message
        predlm = train_msg_new[bestK].lower()
        givenlm = test_msg[i].lower()
        prediff = train_diff_new[bestK]
        final_list_bleu[0].append(test_diff[i].replace('<nl>','\n'))
        final_list_bleu[1].append(givenlm)
        final_list_bleu[2].append(prediff.replace('<nl>','\n'))
        final_list_bleu[3].append(predlm)
        blue_score = sentence_bleu(references=[givenlm.split()], hypothesis=predlm.split(),smoothing_function=chencherry.method5)
        final_list_bleu[4].append(blue_score)
        bleu_scores.append(blue_score)
        x = 4
        
        for j in topK_index_cos:
            final_list_bleu[x].append(train_diff_new[j])
            final_list_bleu[x+1].append(train_msg_new[j])
            x=x+2
        
        
    with open('nearest_diff_topk_100jac_20cos.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(final_list_bleu) 
                    
    print('Average of blue scores:', sum(bleu_scores) / len(bleu_scores) * 100)
    print('size of test data = ', len(bleu_scores))
                    
