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
        modalities: 模态组合 (如 'LA', 'L', 'A')
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
    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator = translator.cuda() if hyp_params.use_cuda else translator
            translator_optimizer = getattr(optim, hyp_params.optim)(
                translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'A':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator = translator.cuda() if hyp_params.use_cuda else translator
            translator_optimizer = getattr(optim, hyp_params.optim)(
                translator.parameters(), lr=hyp_params.lr)
        trans_criterion = getattr(nn, 'MSELoss')()
    else:
        translator = None
        translator_optimizer = None
        trans_criterion = None

    # 初始化主模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 准备设置字典
    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'translator': translator,
        'translator_optimizer': translator_optimizer,
        'trans_criterion': trans_criterion
    }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
# 训练和评估脚本
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    translator = settings['translator']
    translator_optimizer = settings['translator_optimizer']
    trans_criterion = settings['trans_criterion']

    # 各模态序列长度字典，用于分割特征计算一致性损失
    seq_lens = {
        'L': hyp_params.l_len,
        'A': hyp_params.a_len,
        'V': hyp_params.v_len if hasattr(hyp_params, 'v_len') else 0
    }

    def train(model, translator, optimizer, criterion):
        epoch_loss = 0
        model.train()
        if translator is not None:
            translator.train()

        num_batches = hyp_params.n_train // hyp_params.batch_size
        start_time = time.time()

        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):
            # 清零梯度
            model.zero_grad()
            if translator is not None:
                translator.zero_grad()

            # 数据移动到GPU
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            # 应用掩码
            masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
            if hyp_params.dataset == 'meld_senti':
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # meld_sentiment
            else:
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # meld_emotion
            text = text * masks_text
            audio = audio * masks_audio
            batch_size = text.size(0)

            # 分布式训练包装
            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = torch.tensor(0.0, device=text.device)
            fake_a, fake_l = None, None

            # 生成缺失模态（如果需要）
            if translator is not None:
                trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                if hyp_params.modalities == 'L':
                    fake_a = trans_net(text, audio, 'train')
                    trans_loss = trans_criterion(fake_a, audio)
                elif hyp_params.modalities == 'A':
                    fake_l = trans_net(audio, text, 'train')
                    trans_loss = trans_criterion(fake_l, text)

            # 主模型前向传播
            if hyp_params.modalities == 'L':
                preds, gnn_features = net(text, fake_a)
            elif hyp_params.modalities == 'A':
                preds, gnn_features = net(fake_l, audio)
            elif hyp_params.modalities == 'LA':
                preds, gnn_features = net(text, audio)
            else:
                raise ValueError('Unknown modalities type')

            # 计算基础损失
            raw_loss = criterion(preds.transpose(1, 2), labels)

            # 计算跨模态一致性损失
            consistency_weight = 0.1  # 一致性损失权重，可调整
            consistency_loss = compute_consistency_loss(
                gnn_features.permute(1, 0, 2),  # 转换为(seq_len, batch, embed_dim)
                hyp_params.modalities,
                seq_lens
            )
            consistency_loss = consistency_weight * consistency_loss

            # 总损失
            if translator is not None:
                combined_loss = raw_loss + trans_loss + consistency_loss
            else:
                combined_loss = raw_loss + consistency_loss

            # 反向传播
            combined_loss.backward()

            # 更新翻译模型参数
            if translator is not None:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                translator_optimizer.step()

            # 更新主模型参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # 累积损失
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        model.eval()
        if translator is not None:
            translator.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):
                # 数据移动到GPU
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                # 应用掩码
                masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
                if hyp_params.dataset == 'meld_senti':
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)
                else:
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)
                text = text * masks_text
                audio = audio * masks_audio
                batch_size = text.size(0)

                # 分布式评估包装
                net = nn.DataParallel(model) if hyp_params.distribute else model
                fake_a, fake_l = None, None

                # 生成缺失模态（如果需要）
                if translator is not None:
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    if not test:  # 验证阶段
                        if hyp_params.modalities == 'L':
                            fake_a = trans_net(text, audio, 'valid')
                        elif hyp_params.modalities == 'A':
                            fake_l = trans_net(audio, text, 'valid')
                    else:  # 测试阶段
                        if hyp_params.modalities == 'L':
                            fake_a = torch.Tensor().cuda()
                            for i in range(hyp_params.a_len):
                                if i == 0:
                                    fake_a_token = trans_net(text, audio, 'test', eval_start=True)[:, [-1]]
                                else:
                                    fake_a_token = trans_net(text, fake_a, 'test')[:, [-1]]
                                fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                        elif hyp_params.modalities == 'A':
                            fake_l = torch.Tensor().cuda()
                            for i in range(hyp_params.l_len):
                                if i == 0:
                                    fake_l_token = trans_net(audio, text, 'test', eval_start=True)[:, [-1]]
                                else:
                                    fake_l_token = trans_net(audio, fake_l, 'test')[:, [-1]]
                                fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                # 主模型推理
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio)
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio)
                else:
                    raise ValueError('Unknown modalities type')

                # 计算损失
                raw_loss = criterion(preds.transpose(1, 2), labels)
                total_loss += raw_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 打印模型参数数量
    if translator is not None:
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
        val_loss, _, _, _ = evaluate(model, translator, criterion, test=False)
        end = time.time()

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            if translator is not None:
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    if translator is not None:
        translator = load_model(hyp_params, name='TRANSLATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, mask = evaluate(model, translator, criterion, test=True)

    # 评估
    acc = eval_meld(results, truths, mask)
    return acc