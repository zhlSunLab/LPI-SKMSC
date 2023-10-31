
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import json
from val_test_src_1 import process_metrics_test



def train_func(valloader, testloader, epochs, fold,model_save_path, learning_rate, dropout, dim, radius, milestones):
   
    model_1 = torch.load(model_save_path+"model_1.pth",map_location=torch.device('cpu'))
    model_2 = torch.load(model_save_path+"model_2.pth",map_location=torch.device('cpu'))
    model_3 = torch.load(model_save_path+"model_3.pth",map_location=torch.device('cpu'))
    model_4 = torch.load(model_save_path+"model_4.pth",map_location=torch.device('cpu'))
    model_5 = torch.load(model_save_path+"model_5.pth",map_location=torch.device('cpu'))
    model_6 = torch.load(model_save_path+"model_6.pth",map_location=torch.device('cpu'))

    with open(model_save_path+"centers.json",'r') as load_f:
        centers = json.load(load_f)

    c_1 = torch.FloatTensor(centers[0])
    c_2 = torch.FloatTensor(centers[1])
    c_3 = torch.FloatTensor(centers[2])
    c_4 = torch.FloatTensor(centers[3])
    c_5 = torch.FloatTensor(centers[4])
    c_6 = torch.FloatTensor(centers[5])


    # ======================================================================
    # val
    with torch.no_grad():
            model_1.eval()
            model_2.eval()
            model_3.eval()
            model_4.eval()
            model_5.eval()
            model_6.eval()

            print('\nevaluating...')
            loss_every_step_val, label_preds_list_val, label_truth_list_val = [], [], []
            for step, (x, label) in enumerate(tqdm(valloader)):
                p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5 = \
                    torch.as_tensor(x[0], dtype=torch.float32), torch.as_tensor(x[1], dtype=torch.float32), \
                    torch.as_tensor(x[2], dtype=torch.float32), torch.as_tensor(x[3], dtype=torch.float32), \
                    torch.as_tensor(x[4], dtype=torch.float32), torch.as_tensor(x[5], dtype=torch.float32), \
                    torch.as_tensor(x[6], dtype=torch.float32), torch.as_tensor(x[7], dtype=torch.float32), \
                    torch.as_tensor(x[8], dtype=torch.float32), torch.as_tensor(x[9], dtype=torch.float32), \
                    torch.as_tensor(x[10], dtype=torch.float32), torch.as_tensor(x[11], dtype=torch.float32)
                X1 = torch.cat((p_kmer, r_kmer), dim=-1)
                X2 = torch.cat((p_kmer_1, r_kmer_1), dim=-1)
                X3 = torch.cat((p_kmer_2, r_kmer_2), dim=-1)
                X4 = torch.cat((p_kmer_3, r_kmer_3), dim=-1)
                X5 = torch.cat((p_kmer_4, r_kmer_4), dim=-1)
                X6 = torch.cat((p_kmer_5, r_kmer_5), dim=-1)

                label[label == 1] = -1
                label[label == 0] = 1
                label = label[:, 1]
                # foward
                output_1 = model_1(X1)
                output_2 = model_2(X2)
                output_3 = model_3(X3)
                output_4 = model_4(X4)
                output_5 = model_5(X5)
                output_6 = model_6(X6)

                # loss
                dist_1 = torch.sum((output_1 - c_1) ** 2, dim=1)
                losses_1 = torch.where(label == 0, dist_1, ((dist_1 + 1e-8) ** label.float()))
                loss_1 = torch.mean(losses_1)

                dist_2 = torch.sum((output_2 - c_2) ** 2, dim=1)
                losses_2 = torch.where(label == 0, dist_2, ((dist_2 + 1e-8) ** label.float()))
                loss_2 = torch.mean(losses_2)

                dist_3 = torch.sum((output_3 - c_3) ** 2, dim=1)
                losses_3 = torch.where(label == 0, dist_3, ((dist_3 + 1e-8) ** label.float()))
                loss_3 = torch.mean(losses_3)

                dist_4 = torch.sum((output_4 - c_4) ** 2, dim=1)
                losses_4 = torch.where(label == 0, dist_4, ((dist_4 + 1e-8) ** label.float()))
                loss_4 = torch.mean(losses_4)

                dist_5 = torch.sum((output_5 - c_5) ** 2, dim=1)
                losses_5 = torch.where(label == 0, dist_5, ((dist_5 + 1e-8) ** label.float()))
                loss_5 = torch.mean(losses_5)

                dist_6 = torch.sum((output_6 - c_6) ** 2, dim=1)
                losses_6 = torch.where(label == 0, dist_6, ((dist_6 + 1e-8) ** label.float()))
                loss_6 = torch.mean(losses_6)

                dist = dist_1 + dist_2 + dist_3 + dist_4 + dist_5+dist_6
                # append
                label_preds_list_val.append(dist.cpu())
                label_truth_list_val.append(label.cpu())
                loss_every_step_val.append(loss_1 + loss_2 + loss_3 + loss_4 + loss_5+loss_6)

    val_metrics_a_fold = process_metrics_test(loss_every_step_val, label_preds_list_val, label_truth_list_val, radius)

    # ======================================================================
    # test
    with torch.no_grad():
        model_1.eval()
        model_2.eval()
        model_3.eval()
        model_4.eval()
        model_5.eval()
        model_6.eval()

        print('\nevaluating...')
        loss_every_step_test, label_preds_list_test, label_truth_list_test = [], [], []
        for step, (x, label) in enumerate(tqdm(testloader)):
            p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5 = \
                torch.as_tensor(x[0], dtype=torch.float32), torch.as_tensor(x[1], dtype=torch.float32), \
                torch.as_tensor(x[2], dtype=torch.float32), torch.as_tensor(x[3], dtype=torch.float32), \
                torch.as_tensor(x[4], dtype=torch.float32), torch.as_tensor(x[5], dtype=torch.float32), \
                torch.as_tensor(x[6], dtype=torch.float32), torch.as_tensor(x[7], dtype=torch.float32), \
                torch.as_tensor(x[8], dtype=torch.float32), torch.as_tensor(x[9], dtype=torch.float32), \
                torch.as_tensor(x[10], dtype=torch.float32), torch.as_tensor(x[11], dtype=torch.float32)
            X1 = torch.cat((p_kmer, r_kmer), dim=-1)
            X2 = torch.cat((p_kmer_1, r_kmer_1), dim=-1)
            X3 = torch.cat((p_kmer_2, r_kmer_2), dim=-1)
            X4 = torch.cat((p_kmer_3, r_kmer_3), dim=-1)
            X5 = torch.cat((p_kmer_4, r_kmer_4), dim=-1)
            X6 = torch.cat((p_kmer_5, r_kmer_5), dim=-1)

            # if torch.cuda.is_available():
            #     X1, X2, X3, X4, X5,X6, label = X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(), label.cuda()
            label[label == 1] = -1
            label[label == 0] = 1
            label = label[:, 1]
            # foward
            output_1 = model_1(X1)
            output_2 = model_2(X2)
            output_3 = model_3(X3)
            output_4 = model_4(X4)
            output_5 = model_5(X5)
            output_6 = model_6(X6)

            # loss
            dist_1 = torch.sum((output_1 - c_1) ** 2, dim=1)
            losses_1 = torch.where(label == 0, dist_1, ((dist_1 + 1e-8) ** label.float()))
            loss_1 = torch.mean(losses_1)

            dist_2 = torch.sum((output_2 - c_2) ** 2, dim=1)
            losses_2 = torch.where(label == 0, dist_2, ((dist_2 + 1e-8) ** label.float()))
            loss_2 = torch.mean(losses_2)

            dist_3 = torch.sum((output_3 - c_3) ** 2, dim=1)
            losses_3 = torch.where(label == 0, dist_3, ((dist_3 + 1e-8) ** label.float()))
            loss_3 = torch.mean(losses_3)

            dist_4 = torch.sum((output_4 - c_4) ** 2, dim=1)
            losses_4 = torch.where(label == 0, dist_4, ((dist_4 + 1e-8) ** label.float()))
            loss_4 = torch.mean(losses_4)

            dist_5 = torch.sum((output_5 - c_5) ** 2, dim=1)
            losses_5 = torch.where(label == 0, dist_5, ((dist_5 + 1e-8) ** label.float()))
            loss_5 = torch.mean(losses_5)

            dist_6 = torch.sum((output_6 - c_6) ** 2, dim=1)
            losses_6 = torch.where(label == 0, dist_6, ((dist_6 + 1e-8) ** label.float()))
            loss_6 = torch.mean(losses_6)

            dist = dist_1 + dist_2 + dist_3 + dist_4 + dist_5+dist_6
            # append
            label_preds_list_test.append(dist.cpu())
            label_truth_list_test.append(label.cpu())
            loss_every_step_test.append(loss_1 + loss_2 + loss_3 + loss_4 + loss_5+loss_6)
    test_metrics_a_fold = process_metrics_test(loss_every_step_test, label_preds_list_test, label_truth_list_test, radius)
    return val_metrics_a_fold, test_metrics_a_fold

