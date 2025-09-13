import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
# 跨模态一致性损失函数
####################################################################

def compute_consistency_loss(gnn_features, modalities, seq_lens):
    """
    计算跨模态一致性损失，鼓励不同模态特征分布相似

    参数:
        gnn_features: GNN输出的特征 (seq_len, batch_size, embed_dim)
        modalities: 模态组合 (如 'LAV', 'L', 'LA' 等)
        seq_lens: 各模态的序列长度字典
    """
    modal_features = []
    idx = 0

    # 按模态分割特征并进行平均池化
    for modal in modalities:
        if modal in seq_lens and seq_lens[modal] > 0:
            end_idx = idx + seq_lens[modal]
            # 确保索引不越界
            end_idx = min(end_idx, gnn_features.shape[0])
            # 提取当前模态特征并平均
            modal_feat = gnn_features[idx:end_idx].mean(dim=0)  # (batch_size, embed_dim)
            modal_features.append(modal_feat)
            idx = end_idx

    # 计算模态间余弦相似度损失
    if len(modal_features) < 2:
        return torch.tensor(0.0, device=gnn_features.device)

    loss = 0.0
    for i in range(len(modal_features)):
        for j in range(i + 1, len(modal_features)):
            # 计算余弦相似度并取平均
            cos_sim = F.cosine_similarity(modal_features[i], modal_features[j], dim=1).mean()
            loss += (1 - cos_sim)  # 最小化1-余弦相似度

    return loss / len(modal_features)


####################################################################
# 模型初始化
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 初始化翻译模型（如果需要）
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator1 = translator1.cuda()
                translator2 = translator2.cuda()
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'A':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator1 = translator1.cuda()
                translator2 = translator2.cuda()
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'V':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator1 = translator1.cuda()
                translator2 = translator2.cuda()
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LA':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator = translator.cuda()
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator = translator.cuda()
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'AV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            # 添加设备移动
            if hyp_params.use_cuda:
                translator = translator.cuda()
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        else:
            raise ValueError('Unknown modalities type')
        trans_criterion = getattr(nn, 'MSELoss')()
    else:
        translator1 = None
        translator2 = None
        translator = None
        translator1_optimizer = None
        translator2_optimizer = None
        translator_optimizer = None
        trans_criterion = None

    # 初始化主模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 优化器设置（支持BERT特殊优化）
    if hyp_params.use_bert:
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)

    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 准备设置字典
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            settings = {
                'model': model,
                'translator1': translator1,
                'translator2': translator2,
                'translator1_optimizer': translator1_optimizer,
                'translator2_optimizer': translator2_optimizer,
                'trans_criterion': trans_criterion,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler
            }
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            settings = {
                'model': model,
                'translator': translator,
                'translator_optimizer': translator_optimizer,
                'trans_criterion': trans_criterion,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler
            }
        else:
            raise ValueError('Unknown modalities type')
    elif hyp_params.modalities == 'LAV':
        settings = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }
    else:
        raise ValueError('Unknown modalities type')

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
# 训练和评估脚本
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    # 初始化翻译模型相关组件
    if hyp_params.modalities != 'LAV':
        trans_criterion = settings['trans_criterion']
        if hyp_params.modalities in ['L', 'A', 'V']:
            translator1 = settings['translator1']
            translator2 = settings['translator2']
            translator1_optimizer = settings['translator1_optimizer']
            translator2_optimizer = settings['translator2_optimizer']
            translator = (translator1, translator2)
        elif hyp_params.modalities in ['LA', 'LV', 'AV']:
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']
        else:
            raise ValueError('Unknown modalities type')
    else:
        translator = None

    # 各模态序列长度字典，用于分割特征计算一致性损失
    seq_lens = {
        'L': hyp_params.l_len,
        'A': hyp_params.a_len,
        'V': hyp_params.v_len if hasattr(hyp_params, 'v_len') else 0
    }

    def train(model, translator, optimizer, criterion):
        if isinstance(translator, tuple):
            translator1, translator2 = translator

        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities in ['L', 'A', 'V']:
                translator1.train()
                translator2.train()
            elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                translator.train()

        start_time = time.time()

        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)  # 处理标签维度

            # 清零梯度
            model.zero_grad()
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    translator1.zero_grad()
                    translator2.zero_grad()
                elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                    translator.zero_grad()

            # 数据移动到GPU
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)

            # 分布式训练包装
            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = torch.tensor(0.0, device=text.device)
            fake_l, fake_a, fake_v = None, None, None

            # 生成缺失模态（如果需要）
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                    trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                    if hyp_params.modalities == 'L':
                        fake_a = trans_net1(text, audio, 'train')
                        fake_v = trans_net2(text, vision, 'train')
                        trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'A':
                        fake_l = trans_net1(audio, text, 'train')
                        fake_v = trans_net2(audio, vision, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'V':
                        fake_l = trans_net1(vision, text, 'train')
                        fake_a = trans_net2(vision, audio, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                    if hyp_params.modalities == 'LA':
                        fake_v = trans_net((text, audio), vision, 'train')
                        trans_loss = trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'LV':
                        fake_a = trans_net((text, vision), audio, 'train')
                        trans_loss = trans_criterion(fake_a, audio)
                    elif hyp_params.modalities == 'AV':
                        fake_l = trans_net((audio, vision), text, 'train')
                        trans_loss = trans_criterion(fake_l, text)

            # 主模型前向传播（获取预测和GNN特征）
            if hyp_params.modalities == 'L':
                preds, gnn_features = net(text, fake_a, fake_v)
            elif hyp_params.modalities == 'A':
                preds, gnn_features = net(fake_l, audio, fake_v)
            elif hyp_params.modalities == 'V':
                preds, gnn_features = net(fake_l, fake_a, vision)
            elif hyp_params.modalities == 'LA':
                preds, gnn_features = net(text, audio, fake_v)
            elif hyp_params.modalities == 'LV':
                preds, gnn_features = net(text, fake_a, vision)
            elif hyp_params.modalities == 'AV':
                preds, gnn_features = net(fake_l, audio, vision)
            elif hyp_params.modalities == 'LAV':
                preds, gnn_features = net(text, audio, vision)
            else:
                raise ValueError('Unknown modalities type')

            # 处理iemocap数据集的输出格式
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)

            # 计算基础损失
            raw_loss = criterion(preds, eval_attr)

            # 计算跨模态一致性损失
            consistency_weight = 0.1  # 一致性损失权重，可调整
            consistency_loss = compute_consistency_loss(
                gnn_features.permute(1, 0, 2),  # 转换为(seq_len, batch, embed_dim)
                hyp_params.modalities,
                seq_lens
            )
            consistency_loss = consistency_weight * consistency_loss

            # 总损失
            if hyp_params.modalities != 'LAV':
                combined_loss = raw_loss + trans_loss + consistency_loss
            else:
                combined_loss = raw_loss + consistency_loss

            # 反向传播
            combined_loss.backward()

            # 更新翻译模型参数
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    torch.nn.utils.clip_grad_norm_(translator1.parameters(), hyp_params.clip)
                    torch.nn.utils.clip_grad_norm_(translator2.parameters(), hyp_params.clip)
                    translator1_optimizer.step()
                    translator2_optimizer.step()
                elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()

            # 更新主模型参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # 累积损失
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        if isinstance(translator, tuple):
            translator1, translator2 = translator

        model.eval()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities in ['L', 'A', 'V']:
                translator1.eval()
                translator2.eval()
            elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                translator.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # 处理标签维度

                # 数据移动到GPU
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)

                # 分布式评估包装
                net = nn.DataParallel(model) if hyp_params.distribute else model
                fake_l, fake_a, fake_v = None, None, None

                # 生成缺失模态（如果需要）
                if hyp_params.modalities != 'LAV':
                    if not test:  # 验证阶段
                        if hyp_params.modalities in ['L', 'A', 'V']:
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                            if hyp_params.modalities == 'L':
                                fake_a = trans_net1(text, audio, 'valid')
                                fake_v = trans_net2(text, vision, 'valid')
                            elif hyp_params.modalities == 'A':
                                fake_l = trans_net1(audio, text, 'valid')
                                fake_v = trans_net2(audio, vision, 'valid')
                            elif hyp_params.modalities == 'V':
                                fake_l = trans_net1(vision, text, 'valid')
                                fake_a = trans_net2(vision, audio, 'valid')
                        elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                            if hyp_params.modalities == 'LA':
                                fake_v = trans_net((text, audio), vision, 'valid')
                            elif hyp_params.modalities == 'LV':
                                fake_a = trans_net((text, vision), audio, 'valid')
                            elif hyp_params.modalities == 'AV':
                                fake_l = trans_net((audio, vision), text, 'valid')
                    else:  # 测试阶段
                        if hyp_params.modalities in ['L', 'A', 'V']:
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                            if hyp_params.modalities == 'L':
                                # 生成音频模态
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net1(text, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net1(text, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                                # 生成视频模态
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(text, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(text, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'A':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(audio, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(audio, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                # 生成视频模态
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(audio, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(audio, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'V':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(vision, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(vision, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                # 生成音频模态
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net2(vision, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net2(vision, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                        elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                            if hyp_params.modalities == 'LA':
                                # 生成视频模态
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net((text, audio), vision, 'test', eval_start=True)[:,
                                                       [-1]]
                                    else:
                                        fake_v_token = trans_net((text, audio), fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'LV':
                                # 生成音频模态
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net((text, vision), audio, 'test', eval_start=True)[:,
                                                       [-1]]
                                    else:
                                        fake_a_token = trans_net((text, vision), fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            elif hyp_params.modalities == 'AV':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net((audio, vision), text, 'test', eval_start=True)[:,
                                                       [-1]]
                                    else:
                                        fake_l_token = trans_net((audio, vision), fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                # 主模型推理
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a, fake_v)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio, fake_v)
                elif hyp_params.modalities == 'V':
                    preds, _ = net(fake_l, fake_a, vision)
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio, fake_v)
                elif hyp_params.modalities == 'LV':
                    preds, _ = net(text, fake_a, vision)
                elif hyp_params.modalities == 'AV':
                    preds, _ = net(fake_l, audio, vision)
                elif hyp_params.modalities == 'LAV':
                    preds, _ = net(text, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')

                # 处理iemocap数据集的输出格式
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                # 计算损失（评估阶段不包含一致性损失）
                raw_loss = criterion(preds, eval_attr)
                total_loss += raw_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # 打印模型参数数量
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities in ['L', 'A', 'V']:
            mgm_parameter1 = sum([param.nelement() for param in translator1.parameters()])
            mgm_parameter2 = sum([param.nelement() for param in translator2.parameters()])
            mgm_parameter = mgm_parameter1 + mgm_parameter2
        elif hyp_params.modalities in ['LA', 'LV', 'AV']:
            mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')

    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')

    # 训练循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练一个epoch
        train_loss = train(model, translator, optimizer, criterion)

        # 验证
        val_loss, _, _ = evaluate(model, translator, criterion, test=False)
        end = time.time()

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            if hyp_params.modalities in ['L', 'A', 'V']:
                save_model(hyp_params, translator1, name='TRANSLATOR_1')
                save_model(hyp_params, translator2, name='TRANSLATOR_2')
            elif hyp_params.modalities in ['LA', 'LV', 'AV']:
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    if hyp_params.modalities in ['L', 'A', 'V']:
        translator1 = load_model(hyp_params, name='TRANSLATOR_1')
        translator2 = load_model(hyp_params, name='TRANSLATOR_2')
        translator = (translator1, translator2)
    elif hyp_params.modalities in ['LA', 'LV', 'AV']:
        translator = load_model(hyp_params, name='TRANSLATOR')

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, translator, criterion, test=True)

    # 评估
    if hyp_params.dataset in ["mosei_senti", 'mosei-bert']:
        acc = eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset in ['mosi', 'mosi-bert']:
        acc = eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        acc = eval_iemocap(results, truths)
    elif hyp_params.dataset == 'sims':
        acc = eval_sims(results, truths)

    return acc
