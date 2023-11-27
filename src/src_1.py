import math
import os
import time
from functools import reduce
import random
from sklearn import model_selection
import sklearn.metrics
from tqdm import tqdm

import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset
import torch
import math
import string
from functools import reduce



def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            r, p, label = line.split(' ')
            if label == '1':
                pos_pairs.append((r, p))
            elif label == '0':
                neg_pairs.append((r, p))
    return pos_pairs, neg_pairs


def read_data_seq(path):
    seq_dict = {}
    with open(path,'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict

from sklearn.metrics import roc_auc_score

# calculate the 6 metrics of Acc, Sn, Sp, Precision, MCC and AUC
def calc_metrics(y_label, y_proba, radius):
    con_matrix = confusion_matrix(y_label, [1 if x >= radius else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N
    Acc = (TP + TN) / (P + N)
    Pre = (TP) / (TP + FP + 1e-10)
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    AUC = roc_auc_score(y_label, y_proba)

    fen_lei = [1 if x >= radius else 0 for x in y_proba]
    F1 = f1_score(np.array(y_label), np.array(fen_lei))
    return Acc, Sn, Sp, Pre, MCC, AUC, F1

def get_negative_pairs(pro_seqs, rna_seqs, pos_pairs):
    protein_name = pro_seqs.keys()
    list1 = []
    for i in protein_name:
        list1.append(i)
    protein_name = list1

    rna_name = rna_seqs.keys()
    list2 = []
    for i in rna_name:
        list2.append(i)
    rna_name = list2

    neg_pairs = []
    for r in rna_name:
        for p in protein_name:
            neg_pairs.append((r, p))

    index_list = []
    for i, value_p in enumerate(neg_pairs):
        if value_p in pos_pairs:
            index_list.append(i)

    for idx in reversed(index_list):
        del neg_pairs[idx]
    return neg_pairs


def load_data(data_set,DATA_BASE_PATH):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pos_pairs, _ = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')
    neg_pairs = get_negative_pairs(pro_seqs, rna_seqs, pos_pairs)

    return pos_pairs, neg_pairs, pro_seqs, rna_seqs

def coding_pairs(pairs, pro_seqs, rna_seqs, PE, RE, kind):
    samples = []
    for index, rp in enumerate(tqdm(pairs)):
        if rp[0] in rna_seqs and rp[1] in pro_seqs:
            r_seq = rna_seqs[rp[0]]  # rna sequence
            p_seq = pro_seqs[rp[1]]  # protein sequence

            r_seq, r_seq_1, r_seq_2, r_seq_3, r_seq_4, r_seq_5 = segmented_sequence(r_seq)
            p_seq, p_seq_1, p_seq_2, p_seq_3, p_seq_4, p_seq_5 = segmented_sequence(p_seq)

            p_seq_kmer = PE.encode_protein(p_seq) 
            p_seq_1_kmer = PE.encode_protein(p_seq_1)
            p_seq_2_kmer = PE.encode_protein(p_seq_2)
            p_seq_3_kmer = PE.encode_protein(p_seq_3)
            p_seq_4_kmer = PE.encode_protein(p_seq_4)
            p_seq_5_kmer = PE.encode_protein(p_seq_5)

            r_seq_kmer = RE.encode_RNA(r_seq)
            r_seq_1_kmer = RE.encode_RNA(r_seq_1)
            r_seq_2_kmer = RE.encode_RNA(r_seq_2)
            r_seq_3_kmer = RE.encode_RNA(r_seq_3)
            r_seq_4_kmer = RE.encode_RNA(r_seq_4)
            r_seq_5_kmer = RE.encode_RNA(r_seq_5)

            if p_seq_kmer is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(rp[1], rp))
            elif r_seq_kmer is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(rp[0], rp))
            else:
                samples.append([
                    [r_seq_kmer, r_seq_1_kmer, r_seq_2_kmer, r_seq_3_kmer, r_seq_4_kmer, r_seq_5_kmer],
                    [p_seq_kmer, p_seq_1_kmer, p_seq_2_kmer, p_seq_3_kmer, p_seq_4_kmer, p_seq_5_kmer],
                    kind])# [ [], label]
        else:
            print('Skip pair {} according to sequence dictionary.'.format(rp))
    return samples

def segmented_sequence(seq):
    seq_len = len(seq)
    idx = seq_len//5
    seq_1 = seq[:idx]
    seq_2 = seq[idx:(idx*2)]
    seq_3 = seq[(2*idx):(3*idx)]
    seq_4 = seq[(3*idx):(4*idx)]
    seq_5 = seq[(4*idx):]
    return seq, seq_1, seq_2, seq_3, seq_4, seq_5

def standardization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))


def pre_process_data(samples, samples_pred=None, VECTOR_REPETITION_CNN=1,WINDOW_P_UPLIMIT=3, WINDOW_R_UPLIMIT=4):
    # samples: r, p, label
    r_kmer = np.array([x[0][0] for x in samples])
    r_kmer_1 = np.array([x[0][1] for x in samples])
    r_kmer_2 = np.array([x[0][2] for x in samples])
    r_kmer_3 = np.array([x[0][3] for x in samples])
    r_kmer_4 = np.array([x[0][4] for x in samples])
    r_kmer_5 = np.array([x[0][5] for x in samples])

    p_kmer = np.array([x[1][0] for x in samples])
    p_kmer_1 = np.array([x[1][1] for x in samples])
    p_kmer_2 = np.array([x[1][2] for x in samples])
    p_kmer_3 = np.array([x[1][3] for x in samples])
    p_kmer_4 = np.array([x[1][4] for x in samples])
    p_kmer_5 = np.array([x[1][5] for x in samples])

    y_samples = np.array([x[2] for x in samples])

    r_kmer, _ = standardization(r_kmer)
    r_kmer_1, _ = standardization(r_kmer_1)
    r_kmer_2, _ = standardization(r_kmer_2)
    r_kmer_3, _ = standardization(r_kmer_3)
    r_kmer_4, _ = standardization(r_kmer_4)
    r_kmer_5, _ = standardization(r_kmer_5)

    p_kmer, _ = standardization(p_kmer)
    p_kmer_1, _ = standardization(p_kmer_1)
    p_kmer_2, _ = standardization(p_kmer_2)
    p_kmer_3, _ = standardization(p_kmer_3)
    p_kmer_5, _ = standardization(p_kmer_5)

    r_kmer=np.expand_dims(r_kmer, axis=1)
    r_kmer_1=np.expand_dims(r_kmer_1, axis=1)
    r_kmer_2=np.expand_dims(r_kmer_2, axis=1)
    r_kmer_3=np.expand_dims(r_kmer_3, axis=1)
    r_kmer_4=np.expand_dims(r_kmer_4, axis=1)
    r_kmer_5=np.expand_dims(r_kmer_5, axis=1)

    p_kmer=np.expand_dims(p_kmer, axis=1)
    p_kmer_1=np.expand_dims(p_kmer_1, axis=1)
    p_kmer_2=np.expand_dims(p_kmer_2, axis=1)
    p_kmer_3=np.expand_dims(p_kmer_3, axis=1)
    p_kmer_4=np.expand_dims(p_kmer_4, axis=1)
    p_kmer_5=np.expand_dims(p_kmer_5, axis=1)

    return p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4,p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4,r_kmer_5, y_samples


def test_partitioning(test, positive_sample_number, seed=0):
    i = 0
    for value in test:
        if value <= positive_sample_number - 1:
            i += 1
    pos_test = test[:i]
    length = len(pos_test)
    print('test set positive samples number : {}'.format(length))

    neg_test = test[i:]
    np.random.seed(seed)
    np.random.shuffle(neg_test)
    neg_test = neg_test[:length]
    print('test set negative samples number : {}'.format(len(neg_test)))
    test = pos_test + neg_test
    return test


def path_init(root, filename):
    DATA_SET = filename
    DATA_BASE_PATH = root + '/data/'
    RESULT_BASE_PATH = root + '/result/'
    TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"

    print("Dataset: {}".format(DATA_SET))
    # result save path
    result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + time.strftime(TIME_FORMAT, time.localtime()) + "/"
    model_save_path = result_save_path+'models/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    return DATA_SET, DATA_BASE_PATH, RESULT_BASE_PATH, TIME_FORMAT,result_save_path, model_save_path

def my_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def paras_to_txt(argsDict,out):
    out.writelines('------------------ Parameters ------------------' + '\n')
    for eachArg, value in argsDict.items():
        if eachArg is 'epoch1':
            out.writelines('\ntraining config :\n')
        elif eachArg is 'n_heads':
            out.writelines('\nstructure config :\n')
        elif eachArg is 'optimizer':
            out.writelines('\noptimizer config :\n')

        out.writelines(eachArg + ' : ' + str(value) + '\n')

    return out



class Dataset_(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return [self.X[0][item], self.X[1][item], self.X[2][item], self.X[3][item], self.X[4][item], self.X[5][item],
                self.X[6][item], self.X[7][item], self.X[8][item], self.X[9][item], self.X[10][item], self.X[11][item]], self.Y[item]

class Dataset_1(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return [self.X[item],self.Y[item]]


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'
    structs = 'hec'

    element_number = 7
    # number of structure kind
    struct_kind = 3

    # clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
    pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    def __init__(self, WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.WINDOW_P_STRUCT_UPLIMIT = WINDOW_P_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_P_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        # table for amino acid clusters
        self.transtable = str.maketrans(self.pro_intab, self.pro_outtab)

    def encode_protein(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error', 'Error'
        result = []
        offset = 0
        # P2_mer_result, P3_mer_result = [], []
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            vec = vec / vec.max()
            if K == 3:
                P3_mer_result = vec
        return P3_mer_result

# encoder for RNA sequence
class RNAEncoder:
    elements = 'AUCG'
    structs = '.('

    element_number = 4
    struct_kind = 2

    def __init__(self, WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_R_UPLIMIT = WINDOW_R_UPLIMIT
        self.WINDOW_R_STRUCT_UPLIMIT = WINDOW_R_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_R_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

    def encode_RNA(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error', 'Error'
        result = []
        offset = 0
        # R3_mer_result, R4_mer_result = [], []
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            vec = vec / vec.max()  
            if K == 4:
                R4_mer_result = vec
        return R4_mer_result


#=======================================================
#=======================================================

def p_metrics(metrics):
    mean_m = np.mean(np.array(metrics), axis = 0)

    temp = ''
    for i in metrics:
        str1 = str(i) + '\n'
        temp+=str1
    temp+='mean :\n' + str(mean_m.tolist())
    return temp

def val_partitioning(train_raw, positive_sample_number, seed=0):
    pos_samples = []
    neg_samples = []

    for value in train_raw:
        if value <= positive_sample_number - 1:
            pos_samples.append(value)
    
    for value in train_raw:
        if value > positive_sample_number - 1:
            neg_samples.append(value)
    
    pos_train, pos_val = model_selection.train_test_split(pos_samples, test_size=0.125, random_state=seed)
    neg_train, neg_val = model_selection.train_test_split(neg_samples, test_size=0.125, random_state=seed)
    
    np.random.seed(seed)
    np.random.shuffle(neg_val)
    neg_val = neg_val[:len(pos_val)]
    val = pos_val + neg_val
    train = pos_train + neg_train

    print(f"train length {len(train)}")
    print(train[:20],train[-20:])
    print(f"val length {len(val)}")
    print(val[:20],val[-20:])
    return train, val

def process_metrics_test(loss_every_step_test, label_preds_list_test, label_truth_list_test, radius):
    label_preds_test = torch.cat(label_preds_list_test, dim=0)
    label_truth_test = torch.cat(label_truth_list_test, dim=0)
    loss_one_epoch_mean_test=torch.mean(torch.FloatTensor(loss_every_step_test))
    # label turn back
    label_truth_test[label_truth_test == 1] = 0
    label_truth_test[label_truth_test == -1] = 1
    # calculate
    metrics_one_epoch_test = calc_metrics(label_truth_test, label_preds_test, radius)
    print('\ntesting :')
    print('test metrics:\n', metrics_one_epoch_test,'\n')
    return metrics_one_epoch_test