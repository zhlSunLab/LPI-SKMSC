
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F

from src.src_1 import process_metrics_test
from src.src_3 import *


def AE_pretain(trainloader, AE_pretrain_epochs, dropout):
    model_1 = AE_1(l=599, dropout=dropout)
    model_2 = AE_2(l=599, dropout=dropout)
    model_3 = AE_3(l=599, dropout=dropout)
    model_4 = AE_4(l=599, dropout=dropout)
    model_5 = AE_5(l=599, dropout=dropout)
    model_6 = AE_6(l=599, dropout=dropout)

    opt_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)
    opt_2 = torch.optim.Adam(model_2.parameters(), lr=0.001)
    opt_3 = torch.optim.Adam(model_3.parameters(), lr=0.001)
    opt_4 = torch.optim.Adam(model_4.parameters(), lr=0.001)
    opt_5 = torch.optim.Adam(model_5.parameters(), lr=0.001)
    opt_6 = torch.optim.Adam(model_6.parameters(), lr=0.001)

    loss_func = torch.nn.MSELoss()
    model_1.train()
    model_2.train()
    model_3.train()
    model_4.train()
    model_5.train()
    model_6.train()

    print('protein AE pretraining ...')
    for epoch in range(AE_pretrain_epochs):
        loss_list = []
        print('\nepoch {}  pretraining...'.format(epoch))
        for step, (x, label) in enumerate(tqdm(trainloader)):
            p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4,p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4,r_kmer_5 = \
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

            if torch.cuda.is_available():
                X1, X2, X3, X4, X5,X6, model_1, model_2, model_3, model_4, model_5,model_6 = \
                    X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(),\
                    model_1.cuda(), model_2.cuda(), model_3.cuda(), model_4.cuda(), model_5.cuda(),model_6.cuda()
            opt_1.zero_grad()
            opt_2.zero_grad()
            opt_3.zero_grad()
            opt_4.zero_grad()
            opt_5.zero_grad()
            opt_6.zero_grad()

            # forward
            output_1 = model_1(X1)
            output_2 = model_2(X2)
            output_3 = model_3(X3)
            output_4 = model_4(X4)
            output_5 = model_5(X5)
            output_6 = model_6(X6)

            # loss
            loss_1 = loss_func(output_1, X1)
            loss_2 = loss_func(output_2, X2)
            loss_3 = loss_func(output_3, X3)
            loss_4 = loss_func(output_4, X4)
            loss_5 = loss_func(output_5, X5)
            loss_6 = loss_func(output_6, X6)

            # backward
            loss_1.backward()
            loss_2.backward()
            loss_3.backward()
            loss_4.backward()
            loss_5.backward()
            loss_6.backward()

            opt_1.step()
            opt_2.step()
            opt_3.step()
            opt_4.step()
            opt_5.step()
            opt_6.step()

            loss_list.append(loss_1 + loss_2 + loss_3 + loss_4 + loss_5+loss_6)
        mean_loss = torch.mean(torch.FloatTensor(loss_list))
        print('epoch: {} | mean loss: {}'.format(epoch, mean_loss))

    return model_1, model_2, model_3, model_4, model_5, model_6


def train_func(model_AE_1, model_AE_2, model_AE_3, model_AE_4, model_AE_5,model_AE_6, trainloader, valloader, testloader, epochs, fold,
               model_save_path, learning_rate, dropout, dim, radius, milestones):
    # model
    model_1 = EN_1(599, dropout)
    model_2 = EN_2(599, dropout)
    model_3 = EN_3(599, dropout)
    model_4 = EN_4(599, dropout)
    model_5 = EN_5(599, dropout)
    model_6 = EN_6(599, dropout)

    if torch.cuda.is_available():
        model_1, model_2, model_3, model_4, model_5,model_6 = model_1.cuda(), model_2.cuda(), model_3.cuda(), model_4.cuda(), model_5.cuda(), model_6.cuda()
    opt_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
    opt_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
    opt_3 = torch.optim.Adam(model_3.parameters(), lr=learning_rate)
    opt_4 = torch.optim.Adam(model_4.parameters(), lr=learning_rate)
    opt_5 = torch.optim.Adam(model_5.parameters(), lr=learning_rate)
    opt_6 = torch.optim.Adam(model_6.parameters(), lr=learning_rate)

    sch1 = torch.optim.lr_scheduler.MultiStepLR(opt_1,milestones=milestones, gamma=0.1,)
    sch2 = torch.optim.lr_scheduler.MultiStepLR(opt_2,milestones=milestones, gamma=0.1,)
    sch3 = torch.optim.lr_scheduler.MultiStepLR(opt_3,milestones=milestones, gamma=0.1,)
    sch4 = torch.optim.lr_scheduler.MultiStepLR(opt_4,milestones=milestones, gamma=0.1,)
    sch5 = torch.optim.lr_scheduler.MultiStepLR(opt_5,milestones=milestones, gamma=0.1,)
    sch6 = torch.optim.lr_scheduler.MultiStepLR(opt_6,milestones=milestones, gamma=0.1,)

    # center c
    c_1, c_2, c_3, c_4, c_5,c_6, model_1, model_2, model_3, model_4, model_5,model_6 = \
        init_center_c(model_AE_1, model_AE_2, model_AE_3, model_AE_4, model_AE_5,model_AE_6,
                      model_1, model_2, model_3, model_4, model_5,model_6,trainloader, dim)

    model_1.train()
    model_2.train()
    model_3.train()
    model_4.train()
    model_5.train()
    model_6.train()

    # model save path
    model_save_path = model_save_path + 'fold_{}/'.format(fold)
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(epochs):
        print('\nepoch {} training...'.format(epoch))
        loss_every_step_train, label_preds_list_train, label_truth_list_train = [], [], []
        
        if epoch <5:
            model_1.train()
            model_2.train()
            model_3.train()
            model_4.train()
            model_5.train()
            model_6.train()
        # After epoch reaches 5, use model.eval() to turn off bn and dropout to make the training stable.
        elif epoch >= 5: 
            model_1.eval()
            model_2.eval()
            model_3.eval()
            model_4.eval()
            model_5.eval()
            model_6.eval()

        for step, (x, label) in enumerate(tqdm(trainloader)):
            p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4,p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4,r_kmer_5 = \
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

            if torch.cuda.is_available():
                X1, X2, X3, X4, X5,X6, label = X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(), label.cuda()

            label[label == 1] = -1
            label[label == 0] = 1
            label = label[:, 1]
            # grad to zero
            opt_1.zero_grad()
            opt_2.zero_grad()
            opt_3.zero_grad()
            opt_4.zero_grad()
            opt_5.zero_grad()
            opt_6.zero_grad()

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
            loss_every_step_train.append(loss_1 + loss_2 + loss_3 + loss_4 + loss_5+loss_6)
            label_preds_list_train.append(dist.cpu())
            label_truth_list_train.append(label.cpu())
            # backward
            loss_1.backward()
            loss_2.backward()
            loss_3.backward()
            loss_4.backward()
            loss_5.backward()
            loss_6.backward()

            opt_1.step()
            opt_2.step()
            opt_3.step()
            opt_4.step()
            opt_5.step()
            opt_6.step()

        sch1.step()
        sch2.step()
        sch3.step()
        sch4.step()
        sch5.step()
        sch6.step()
        print('train loss: ', torch.mean(torch.FloatTensor(loss_every_step_train)))


    # ======================================================================
    # all epochs finished, validation
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

            if torch.cuda.is_available():
                X1, X2, X3, X4, X5,X6, label = X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(), label.cuda()
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

            if torch.cuda.is_available():
                X1, X2, X3, X4, X5,X6, label = X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(), label.cuda()
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


def init_center_c(model_AE_1, model_AE_2, model_AE_3, model_AE_4, model_AE_5,model_AE_6,
                  model_1, model_2, model_3, model_4, model_5,model_6, trainloader, dim):
    # get encoder
    model_1_dict = model_1.state_dict()
    model_AE_1_dict = model_AE_1.state_dict()

    model_2_dict = model_2.state_dict()
    model_AE_2_dict = model_AE_2.state_dict()

    model_3_dict = model_3.state_dict()
    model_AE_3_dict = model_AE_3.state_dict()

    model_4_dict = model_4.state_dict()
    model_AE_4_dict = model_AE_4.state_dict()

    model_5_dict = model_5.state_dict()
    model_AE_5_dict = model_AE_5.state_dict()

    model_6_dict = model_6.state_dict()
    model_AE_6_dict = model_AE_6.state_dict()

    model_AE_1_dict = {k: v for k, v in model_AE_1_dict.items() if k in model_1_dict}
    model_AE_2_dict = {k: v for k, v in model_AE_2_dict.items() if k in model_2_dict}
    model_AE_3_dict = {k: v for k, v in model_AE_3_dict.items() if k in model_3_dict}
    model_AE_4_dict = {k: v for k, v in model_AE_4_dict.items() if k in model_4_dict}
    model_AE_5_dict = {k: v for k, v in model_AE_5_dict.items() if k in model_5_dict}
    model_AE_6_dict = {k: v for k, v in model_AE_6_dict.items() if k in model_6_dict}

    model_1_dict.update(model_AE_1_dict)
    model_1.load_state_dict(model_1_dict)

    model_2_dict.update(model_AE_2_dict)
    model_2.load_state_dict(model_2_dict)

    model_3_dict.update(model_AE_3_dict)
    model_3.load_state_dict(model_3_dict)

    model_4_dict.update(model_AE_4_dict)
    model_4.load_state_dict(model_4_dict)

    model_5_dict.update(model_AE_5_dict)
    model_5.load_state_dict(model_5_dict)

    model_6_dict.update(model_AE_6_dict)
    model_6.load_state_dict(model_6_dict)

    # get init center c
    c_1 = torch.zeros(dim)
    c_2 = torch.zeros(dim)
    c_3 = torch.zeros(dim)
    c_4 = torch.zeros(dim)
    c_5 = torch.zeros(dim)
    c_6 = torch.zeros(dim)

    n_samples = 0
    eps = 0.1

    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()

    with torch.no_grad():
        print('init center c')
        for step, (x, label) in enumerate(tqdm(trainloader)):
            p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4,p_kmer_5, r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4,r_kmer_5 = \
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

            if torch.cuda.is_available():
                X1, X2, X3, X4, X5,X6, label, c_1, c_2, c_3, c_4, c_5,c_6 = \
                    X1.cuda(), X2.cuda(), X3.cuda(), X4.cuda(), X5.cuda(),X6.cuda(), label.cuda(), \
                                                   c_1.cuda(), c_2.cuda(), c_3.cuda(), c_4.cuda(), c_5.cuda(),c_6.cuda()

            output_1 = model_1(X1)
            output_2 = model_2(X2)
            output_3 = model_3(X3)
            output_4 = model_4(X4)
            output_5 = model_5(X5)
            output_6 = model_6(X6)

            n_samples += output_1.shape[0]
            c_1 += torch.sum(output_1, dim=0)
            c_2 += torch.sum(output_2, dim=0)
            c_3 += torch.sum(output_3, dim=0)
            c_4 += torch.sum(output_4, dim=0)
            c_5 += torch.sum(output_5, dim=0)
            c_6 += torch.sum(output_6, dim=0)

    c_1 /= n_samples
    c_2 /= n_samples
    c_3 /= n_samples
    c_4 /= n_samples
    c_5 /= n_samples
    c_6 /= n_samples

    c_1[(abs(c_1) < eps) & (c_1 < 0)] = -eps
    c_1[(abs(c_1) < eps) & (c_1 > 0)] = eps
    c_2[(abs(c_2) < eps) & (c_2 < 0)] = -eps
    c_2[(abs(c_2) < eps) & (c_2 > 0)] = eps
    c_3[(abs(c_3) < eps) & (c_3 < 0)] = -eps
    c_3[(abs(c_3) < eps) & (c_3 > 0)] = eps
    c_4[(abs(c_4) < eps) & (c_4 < 0)] = -eps
    c_4[(abs(c_4) < eps) & (c_4 > 0)] = eps
    c_5[(abs(c_5) < eps) & (c_5 < 0)] = -eps
    c_5[(abs(c_5) < eps) & (c_5 > 0)] = eps
    c_6[(abs(c_6) < eps) & (c_6 < 0)] = -eps
    c_6[(abs(c_6) < eps) & (c_6 > 0)] = eps

    return c_1, c_2, c_3, c_4, c_5,c_6, model_1, model_2, model_3, model_4, model_5,model_6