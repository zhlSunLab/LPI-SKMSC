
import numpy as np
import torch.utils.data

import argparse
from keras.utils import np_utils
from torch.utils.data import Dataset
from src.src_1 import *
from src.src_2 import *

# args
def parse_args():

    parser = argparse.ArgumentParser()
    # init
    parser.add_argument('-r', '--root', default='./', type=str)
    parser.add_argument('-f', '--filename', default='RPI1847', type=str)#RPI1847,RPI7317,RPI488

    # training config
    parser.add_argument('-e1', '--epochs', default=20, type=int)
    parser.add_argument('--AE_pretrian_epochs', default=15, type=int)
    parser.add_argument('-b', '--batchsize', default=64, type=int)
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--radius', default=0.5, type=float)
    parser.add_argument('--ms1', default=15, type=int)
    parser.add_argument('--ms2', default=18, type=int)

    # structure config
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # optimizer config
    parser.add_argument('--learning_rate', default=0.001, type=float)

    args = parser.parse_args()
    return args

def main(args):
    milestones = [args.ms1, args.ms2]
    # seed
    my_setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # path
    DATA_SET, DATA_BASE_PATH, RESULT_BASE_PATH, TIME_FORMAT, result_save_path, model_save_path = path_init(args.root, args.filename)
    # hyper params
    WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT = 3, 3
    WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT = 4, 4
    VECTOR_REPETITION_CNN = 1
    K_FOLD = 5

    out = open(result_save_path + 'result.txt', 'w')
    out.write('LPI_SKMSC.py\n')
    argsDict = args.__dict__
    out = paras_to_txt(argsDict, out)

    '''==================================================='''
    # data
    # read rna-protein pairs and sequences from data files
    pos_pairs, neg_pairs, pro_seqs, rna_seqs = load_data(DATA_SET,DATA_BASE_PATH)
    # sequence encoder instance
    PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, True, VECTOR_REPETITION_CNN)
    RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, True, VECTOR_REPETITION_CNN)

    print("Coding positive protein-rna pairs.\n")
    samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, PE, RE, kind=1)
    positive_sample_number = len(samples)
    print("Coding negative protein-rna pairs.\n")
    samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, PE, RE, kind=0)
    negative_sample_number = len(samples) - positive_sample_number
    samples_len = len(samples)

    p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4,p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4,r_kmer_5, y = pre_process_data(samples=samples,
                            VECTOR_REPETITION_CNN=VECTOR_REPETITION_CNN,
                            WINDOW_P_UPLIMIT=WINDOW_P_UPLIMIT,WINDOW_R_UPLIMIT=WINDOW_R_UPLIMIT)

    # sample numbers for the positive and the negative
    print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
    out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))


    val_metrics_all_fold = []
    test_metrics_all_fold = []

    for fold in range(args.k_fold):
        train_raw = [i for i in range(int(samples_len)) if i % K_FOLD != fold] 
        train, val = val_partitioning(train_raw, positive_sample_number, seed=args.seed)

        test = [i for i in range(int(samples_len)) if i % K_FOLD == fold]
        test = test_partitioning(test, positive_sample_number, seed=args.seed)
        print(test[:20], test[-20:])

        # generate train and test data
        X_train = [p_kmer[train], p_kmer_1[train], p_kmer_2[train], p_kmer_3[train], p_kmer_4[train],p_kmer_5[train],
                   r_kmer[train], r_kmer_1[train], r_kmer_2[train], r_kmer_3[train], r_kmer_4[train],r_kmer_5[train]]
        X_val = [p_kmer[val], p_kmer_1[val], p_kmer_2[val], p_kmer_3[val], p_kmer_4[val],p_kmer_5[val],
                   r_kmer[val], r_kmer_1[val], r_kmer_2[val], r_kmer_3[val], r_kmer_4[val],r_kmer_5[val]]
        X_test = [p_kmer[test], p_kmer_1[test], p_kmer_2[test], p_kmer_3[test], p_kmer_4[test],p_kmer_5[test],
                  r_kmer[test], r_kmer_1[test], r_kmer_2[test], r_kmer_3[test], r_kmer_4[test],r_kmer_5[test]]

        y_train = np_utils.to_categorical(y[train], 2)
        y_val = np_utils.to_categorical(y[val], 2)
        y_test = np_utils.to_categorical(y[test], 2)
        print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))

        # =================================================================
        train_data = Dataset_(X_train, y_train)
        val_data = Dataset_(X_val, y_val)
        test_data = Dataset_(X_test, y_test)

        trainloader = torch.utils.data.DataLoader(train_data,
                                                  num_workers=args.num_workers,
                                                  batch_size=args.batchsize,
                                                  shuffle=True,
                                                  )
        valloader = torch.utils.data.DataLoader(val_data,
                                                num_workers=args.num_workers,
                                                batch_size=args.batchsize,
                                                shuffle=False,
                                                )
        testloader = torch.utils.data.DataLoader(test_data,
                                                  num_workers=args.num_workers,
                                                  batch_size=args.batchsize,
                                                  shuffle=False,
                                                )
        # AE pretrain
        model_AE_1, model_AE_2, model_AE_3, model_AE_4, model_AE_5, model_AE_6 = AE_pretain(trainloader, args.AE_pretrian_epochs, args.dropout)

        # train and val, calculate metrics
        val_metrics_a_fold, test_metrics_a_fold = train_func(model_AE_1, model_AE_2, model_AE_3, model_AE_4, model_AE_5,model_AE_6,
                                                    trainloader, valloader, testloader,
                                                    args.epochs, args.k_fold, model_save_path,
                                                    args.learning_rate, args.dropout, args.dim,args.radius,milestones)
        # record metrics
        val_metrics_all_fold.append(val_metrics_a_fold)
        test_metrics_all_fold.append(test_metrics_a_fold)

    # all folds finish
    val_metrics = p_metrics(val_metrics_all_fold)
    out.write('\n\n val performance:\n' + val_metrics)

    test_metrics = p_metrics(test_metrics_all_fold)
    out.write('\n\n test performance:\n' + test_metrics)
    print('\n test performance:\n', test_metrics,'\n')

    out.flush()
    out.close()


if __name__ == '__main__':
    main(parse_args())




