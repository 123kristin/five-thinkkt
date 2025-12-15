import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

# 导入设备配置，将从外部传入
# import os

# 删除 get_device 和 device 全局变量
# def get_device():
#     gpu_id = os.environ.get('CURRENT_GPU_ID', '1')
#     return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
# device = get_device()

def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    if model_name in ["atdkt", "simplekt", "stablekt", "bakt_time", "sparsekt", "cskt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        loss1 = binary_cross_entropy(y.double(), t.double())
        
        # 检查损失值
        if torch.isnan(loss1) or torch.isinf(loss1):
            print(f"[cal_loss] 警告: loss1为NaN或Inf，使用默认值")
            loss1 = torch.tensor(0.0, device=y.device, requires_grad=True)

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
    elif model_name in ["rekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        loss = binary_cross_entropy(y.double(), t.double())
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[cal_loss] 警告: 损失值为NaN或Inf，使用默认值")
            loss = torch.tensor(0.0, device=y.device, requires_grad=True)
    
    elif model_name in ["ukt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        loss1 = binary_cross_entropy(y.double(), t.double())
        
        # 检查损失值
        if torch.isnan(loss1) or torch.isinf(loss1):
            print(f"[cal_loss] 警告: loss1为NaN或Inf，使用默认值")
            loss1 = torch.tensor(0.0, device=y.device, requires_grad=True)
        
        if model.use_CL:
            loss2 = ys[1]
            loss1 = loss1 + model.cl_weight * loss2
        loss = loss1

    elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        loss = binary_cross_entropy(y.double(), t.double())
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[cal_loss] 警告: 损失值为NaN或Inf，使用默认值")
            loss = torch.tensor(0.0, device=y.device, requires_grad=True)
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","lefokt_akt", "dtransformer", "fluckt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[cal_loss] 警告: 损失值为NaN或Inf，使用默认值")
            loss = torch.tensor(0.0, device=y.device, requires_grad=True)
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        
        # 检查数值稳定性
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[cal_loss] 警告: y包含NaN或Inf值")
            y = torch.zeros_like(y)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[cal_loss] 警告: t包含NaN或Inf值")
            t = torch.zeros_like(t)
        
        # 确保y在[0,1]范围内
        y = torch.clamp(y, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)
        
        criterion = nn.BCELoss(reduction='none')        
        loss = criterion(y, t).sum()
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[cal_loss] 警告: 损失值为NaN或Inf，使用默认值")
            loss = torch.tensor(0.0, device=y.device, requires_grad=True)
    
    return loss


def model_forward(model, data, rel=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(next(model.parameters()).device), dcur["cseqs"].to(next(model.parameters()).device), dcur["rseqs"].to(next(model.parameters()).device), dcur["tseqs"].to(next(model.parameters()).device),dcur["sdseqs"].to(next(model.parameters()).device),dcur["qdseqs"].to(next(model.parameters()).device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(next(model.parameters()).device), dcur["shft_cseqs"].to(next(model.parameters()).device), dcur["shft_rseqs"].to(next(model.parameters()).device), dcur["shft_tseqs"].to(next(model.parameters()).device),dcur["shft_sdseqs"].to(next(model.parameters()).device),dcur["shft_qdseqs"].to(next(model.parameters()).device)
    else:
        q, c, r, t = dcur["qseqs"].to(next(model.parameters()).device), dcur["cseqs"].to(next(model.parameters()).device), dcur["rseqs"].to(next(model.parameters()).device), dcur["tseqs"].to(next(model.parameters()).device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(next(model.parameters()).device), dcur["shft_cseqs"].to(next(model.parameters()).device), dcur["shft_rseqs"].to(next(model.parameters()).device), dcur["shft_tseqs"].to(next(model.parameters()).device)
    m, sm = dcur["masks"].to(next(model.parameters()).device), dcur["smasks"].to(next(model.parameters()).device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "stablekt", "sparsekt", "cskt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]
    elif model_name in ["ukt"]:
        if model.use_CL != 0 :
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:,1:],sim,y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:,1:], y2, y3]
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lefokt_akt", "fluckt"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(next(model.parameters()).device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        if model_name == "crkt":
            # crkt返回(y, total_loss, main_loss, contrastive_loss)
            y, loss = model.train_one_step(data)
        elif model_name == "thinkkt":
            # thinkkt返回(y, loss)
            y, loss = model.train_one_step(data)
        else:
            y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 
    
    else:
        y,loss = model.train_one_step(data)

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss,temp
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            
            # 设置模型为训练模式
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()  # 这里会设置model.training = True
            
            # 前向传播和损失计算
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel)
            elif model.model_name in ["ukt"] and model.use_CL != 0:
                loss,temp = model_forward(model, data)
            else:
                loss = model_forward(model, data)
            
            # 反向传播和优化
            opt.zero_grad()
            loss.backward()
            
            # 梯度裁剪（针对特定模型）
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
                
            loss_mean.append(loss.detach().cpu().numpy())
            
            # 调试输出（针对特定模型）
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        
        # 学习率调度
        if model.model_name=='lpkt':
            scheduler.step()
        
        loss_mean = np.mean(loss_mean)
        
        # 验证阶段
        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)

        # 保存最佳模型
        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        
        # 训练日志输出
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")

        # 早停机制
        # if i - best_epoch >= 10:
        #     break
        if i - best_epoch >= 10:
            break
    
    # 训练结束时保存特征缓存和CoT缓存
    if hasattr(model, 'save_feature_cache'):
        model.save_feature_cache()
    
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch