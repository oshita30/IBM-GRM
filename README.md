# Log Message Generation
Experiments with the log message generation application in the paper, CC2Vec: Distributed Representations of Code Changes (https://arxiv.org/pdf/2003.05620.pdf)

Public Source code of CC2Vec: https://github.com/CC2Vec/CC2Vec

## 1. The following are for files in the "source" subfolder:

- Original implementation of CC2Vec for log message generation:

      $ python lmg_eval.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data]
      
- Modified implementation to align with the description in paper:

      $ python lmg_eval_modified.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK [value of k (int)]

- nearest_diffs.py
 This file calculates the topK patches corresponding to each diff in the test file and stores in a csv, predicted patch is also stored in the csv.
 
       $ python nearest_diffs.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK [value of k (int)] -name_of_csvfile [name of file in which the output is to be stored]
       
- topK_dictionary.py
Stores the BLEU score for different values of K in a dictionary and dumps the pkl file into the current directory, graph can be plotted using this dictionary.

       $ python topK_dictionary.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK [value of k (int)] -lower_limit [lower value of K] -upper_limit [upper value of K] -step [step size between lower and upper limit]
       
## 2. kNN_cosine_BERT.py
This file summarizes the topK messages usind BERT extractive summarization and stores results in a csv file.

       $ python kNN_cosine_BERT.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK_csoine [value of k for cosine similarity(int)] -csv_name [name of file in which the output is to be stored]

## 3. kNN_cosine_jaccard.py
This file replaces BLEU with jaccard similarity in the original implementation and stores the results in a csv.
The csv contains test-diff, test-lm, pred-diff, pred-lm, top20 diffs from training set based on cosine similarity.

       $ python kNN_cosine_jaccard.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK_csoine [value of k for cosine similarity(int)]

## 4. kNN_cosine_editDist.py
Same as kNN_cosine_jaccard, just jaccard replaced by edit_distance.

       $ python kNN_cosine_editDist.py -train_data [path of our training data] -test_data [path of our testing data] -train_cc2ftr_data [path of our code changes features extracted from training data] -test_cc2ftr_data [path of our code changes features extracted from testing data] -topK_csoine [value of k for cosine similarity(int)]
       
