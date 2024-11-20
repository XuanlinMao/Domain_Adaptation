import torch
import torch.optim as optim
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from utils import *
from typing import Literal
import pandas as pd 
import os
import gc
from models import *
from os.path import join
from tqdm import tqdm
from pygod.metric import eval_roc_auc
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')



class SupRunner():
    def __init__(
            self,
            data_src,
            label_src,
            data_tgt,
            label_tgt,
            source_encoder,
            classifier,
            device='cpu'
        ) -> None:
        
        self.data_src = data_src
        self.label_src = label_src
        self.data_tgt = data_tgt
        self.label_tgt = label_tgt

        self.source_encoder = source_encoder
        self.classifier = classifier
        self.device = device


    def cpu(self):
        device = torch.device('cpu')
        self.source_encoder = self.source_encoder.to(device)
        self.classifier = self.classifier.to(device)
        self.data_src, self.label_src = self.data_src.to(device), self.label_src.to(device)
        self.data_tgt, self.label_tgt = self.data_tgt.to(device), self.label_tgt.to(device)
        torch.cuda.empty_cache()


    def train(
            self, 
            lr:float=1e-2, 
            n_epoch:int=1000, 
            early_stopping:bool=True, 
            n_stopping:int=20,
            printing:bool=True
        ) -> None:

        loss_val_best = np.inf
        stops = 0
        self.source_encoder.reset_parameters()
        self.classifier.reset_parameters()
        self.source_encoder = self.source_encoder.to(self.device)
        self.classifier = self.classifier.to(self.device)
        self.data_src, self.label_src = self.data_src.to(self.device), self.label_src.to(self.device)
        self.data_tgt, self.label_tgt = self.data_tgt.to(self.device), self.label_tgt.to(self.device)

        source_optimizer = optim.Adam([{'params':self.source_encoder.parameters(),'lr':lr},
                                    {'params':self.classifier.parameters(),'lr':lr}])
        loss_bce = nn.BCELoss()

        if printing: print("== Start Training ===")
        for epoch in range(n_epoch):
            self.source_encoder.train()
            self.classifier.train()

            source_encodings = self.source_encoder(self.data_src.x,self.data_src.edge_index)
            source_pred = self.classifier(source_encodings)
            loss = loss_bce(source_pred[self.data_src.train_mask,:], self.label_src[self.data_src.train_mask,:])
            source_optimizer.zero_grad()
            loss.backward()
            loss_train = loss.detach().cpu().numpy()
            source_optimizer.step()

            with torch.no_grad():
                self.source_encoder.eval()
                self.classifier.eval()
                loss_val = loss_bce(source_pred[self.data_src.val_mask,:], 
                                    self.label_src[self.data_src.val_mask,:]).detach().cpu().numpy()
            
            torch.cuda.empty_cache()
            if early_stopping:
                if loss_val < loss_val_best:
                    stops = 0
                    loss_val_best = loss_val
                    weights_source_encoder = deepcopy(self.source_encoder.state_dict())
                    weights_classifier = deepcopy(self.classifier.state_dict())
                    epoch_best = epoch
                else:
                    stops += 1
                if stops >= n_stopping:
                    if printing: print(f"=== Early Stopping at epoch {epoch_best}, best loss_val = {loss_val_best} ===")
                    break
            if printing and epoch%10==0: 
                print(f"Epoch: {epoch}, Training loss: {loss_train}, Val loss: {loss_val}")
        
        if early_stopping:
            self.source_encoder.load_state_dict(weights_source_encoder)
            self.classifier.load_state_dict(weights_classifier)
        if printing: print("=== DONE ===")


    def _get_score(
            self,
            part:Literal["train","val","test","source","target"]
        ):
        with torch.no_grad():
            self.source_encoder.eval()
            self.classifier.eval()
            if part=="train":
                score = self.classifier(self.source_encoder(self.data_src.x,self.data_src.edge_index))[self.data_src.train_mask,:]
                y_true = self.label_src[self.data_src.train_mask,:]
            elif part=="val":
                score = self.classifier(self.source_encoder(self.data_src.x,self.data_src.edge_index))[self.data_src.val_mask,:]
                y_true = self.label_src[self.data_src.val_mask,:]
            elif part=="test":
                score = self.classifier(self.source_encoder(self.data_src.x,self.data_src.edge_index))[self.data_src.test_mask,:]
                y_true = self.label_src[self.data_src.test_mask,:]
            elif part=="source":
                score = self.classifier(self.source_encoder(self.data_src.x,self.data_src.edge_index))
                y_true = self.label_src
            else: # target
                score = self.classifier(self.source_encoder(self.data_tgt.x,self.data_tgt.edge_index))
                y_true = self.label_tgt
        return score,y_true


    def performance(
            self,
            part:Literal["train","val","test","source","target"],
            threshold:float=0.5,
            Klist:int=[50,100,200,300],
            printing:bool=True,
            res_return:bool=False
        ) -> None:

        score,y_true = self._get_score(part)
        res = eval_all(y_true, score, threshold, Klist)
        if printing:
            for k,v in res.items():
                print(f"{k}: {v:.4f}")
        if res_return:
            return res
    
    def get_perf_df(
            self,
            threshold:float=0.5,
            Klist:int=[50,100,200,300]
        ):
        res_target = self.performance('target',printing=False,res_return=True,threshold=threshold,Klist=Klist)
        res_train = self.performance('train',printing=False,res_return=True,threshold=threshold,Klist=Klist)
        res_val = self.performance('val',printing=False,res_return=True,threshold=threshold,Klist=Klist)
        df1 = pd.DataFrame(res_train, index=['train'])
        df2 = pd.DataFrame(res_val, index=['validation'])
        df3 = pd.DataFrame(res_target, index=['target'])
        return pd.concat([df1,df2,df3],axis=0)
    
    def save(
        self,
        modulename,
        outdir,
        savedir,
        source
    ):
        torch.save(self.source_encoder.state_dict(), os.path.join(savedir,f'{modulename}_encoder_{source}.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(savedir,f'{modulename}_classifier_{source}.pth'))
        self.get_perf_df().to_csv(os.path.join(outdir,f'{modulename}_perf_{source}.csv'))
    
    def clear(self):
        self.cpu()
        del(self.source_encoder)
        del(self.classifier)
        del(self.data_src)
        del(self.data_tgt)
        del(self.label_src)
        del(self.label_tgt)
        gc.collect()
        torch.cuda.empty_cache()





class DARunner():
    def __init__(self, args) -> None:
        self.args = args
    
    def train(self):
        args = self.args

        device = torch.device(f'cuda:{args.device}')
        set_random_seeds(seed_value=args.seed, device=device)

        encoder = GCN_Encoder(nhids = args.nhids_source_encoder,
                              dropout = args.dropout_source_encoder,
                              with_bn = args.with_bn_source_encoder).to(device)
        classifier = Classifier(nhids = args.nhids_classifier,
                                dropout = args.dropout_classifier,
                                with_bn = args.with_bn_classifier).to(device)
        encoder.load_state_dict(torch.load(join(args.savedir_source, "GCN_encoder_" + args.source + ".pth")))
        classifier.load_state_dict(torch.load(join(args.savedir_source, "GCN_classifier_" + args.source + ".pth")))

        data_tgt, label_tgt = load_data(args.target, if_split=False)
        G = nx.Graph()
        G.add_edges_from(data_tgt.edge_index.t().tolist())
        adj = torch.tensor(nx.to_numpy_array(G, dtype=int)).to(device).float()
        data_tgt, label_tgt = data_tgt.to(device), label_tgt.to(device)

        n_t = data_tgt.x.shape[0]
        n_epoch = args.n_epoch_t
        k = args.k
        # m = args.m
        # r = args.r
        gamma_div = args.gamma_div
        gamma_semantic = args.gamma_semantic
        gamma_hetero = args.gamma_hetero
        epsilon = args.epsilon
        alpha = args.alpha
        lr = args.lr_t
        epoch_encoder = args.epoch_encoder
        epoch_classifier = args.epoch_classifier

        encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=lr)
        classifier_optimizer = optim.Adam(params=classifier.parameters(), lr=lr)

        loss_div_list = []
        loss_hetero_list = []
        loss_semantic_list = []
        auc_list = []


        print("==== Start DA ====")
        for epoch in tqdm(range(n_epoch)):
            if epoch % (epoch_encoder + epoch_classifier) < epoch_encoder:
                for p in encoder.parameters():
                    p.requires_grad = True
                for p in classifier.parameters():
                    p.requires_grad = False
                encoder.train()
                classifier.train()

                emb = encoder(data_tgt.x, data_tgt.edge_index).to(device)
                pred = classifier(emb)
                pred = torch.cat([1-pred,pred],dim=1).to(device)
                cosine_sim = F.normalize(emb, p=2, dim=1) @ F.normalize(emb, p=2, dim=1).T
                indices_k = cosine_sim.topk(k+1, dim=1).indices
                # indices_m = cosine_sim.topk(m+1, dim=1).indices
                del(cosine_sim, emb)
                torch.cuda.empty_cache()

                # adj_cos_k = torch.zeros([n_t,n_t]).to(device)
                # adj_cos_k.scatter_(1, indices_k, 1.)
                # adj_cos_k -= torch.diag(torch.diag(adj_cos_k))
                # adj_cos_m = torch.zeros([n_t,n_t]).to(device)
                # adj_cos_m.scatter_(1, indices_m, 1.)
                # adj_cos_m -= torch.diag(torch.diag(adj_cos_m))
                # adj_cos = adj_cos_k * adj_cos_m

                # adj_cos = torch.zeros([n_t,n_t]).to(device)
                # adj_cos.scatter_(1, indices_k, 1.)
                # adj_cos -= torch.diag(torch.diag(adj_cos))
                # adj_cos[adj_cos == 0.] = r

                indices = torch.cat([torch.arange(indices_k.shape[0]).repeat_interleave(indices_k.shape[1]).to(device).reshape(1,-1), 
                                     indices_k.flatten().reshape(1,-1)], dim=0) # transfer indices_k to edge_idx form
                values = torch.ones(indices.size(1), device=device)
                adj_cos = torch.sparse_coo_tensor(indices, values, size=(n_t, n_t)).to(device)
                adj_cos = adj_cos.coalesce()
                diag_mask = adj_cos._indices()[0] != adj_cos._indices()[1]
                indices = adj_cos._indices()[:, diag_mask]
                values = adj_cos._values()[diag_mask]
                adj_cos = torch.sparse_coo_tensor(indices, values, size=(n_t, n_t)).to(device)
                
                q = torch.tensor([1-alpha, alpha]).to(device)
                p = pred.mean(dim=0)
                
                loss_div = (p * torch.log(p/q)).sum()
                # loss_semantic = - ((adj_cos @ pred) * pred).sum(dim=1).mean()
                loss_semantic = - (torch.sparse.mm(adj_cos, pred) * pred).sum(dim=1).mean()
                loss = gamma_div*loss_div + gamma_semantic*loss_semantic

                encoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()

                loss_div_list.append(loss_div.detach().item())
                loss_semantic_list.append(loss_semantic.detach().item())
                loss_hetero_list.append(None)
            
            else:
                for p in encoder.parameters():
                    p.requires_grad = False
                for p in classifier.parameters():
                    p.requires_grad = True
                encoder.train()
                classifier.train()

                emb = encoder(data_tgt.x, data_tgt.edge_index).to(device)
                pred = classifier(emb)
                pred = torch.clamp(pred, min=epsilon, max=1-epsilon)
                pred_nb = (adj@pred)/adj.sum(dim=1,keepdim=True)
                hetero = (1-F.cosine_similarity(torch.cat([pred,1-pred],dim=1), torch.cat([pred_nb,1-pred_nb],dim=1))).reshape(-1,1).detach()
                loss_hetero = -(hetero*torch.log(pred) + (1-hetero)*torch.log(1-pred)).mean()
                loss = loss_hetero

                classifier_optimizer.zero_grad()
                loss.backward()
                classifier_optimizer.step()
                
                loss_hetero_list.append(loss_hetero.detach().item())
                loss_div_list.append(None)
                loss_semantic_list.append(None)

            for p in encoder.parameters():
                p.requires_grad = False
            for p in classifier.parameters():
                    p.requires_grad = False
            encoder.eval()
            classifier.eval()

            pred = classifier(encoder(data_tgt.x, data_tgt.edge_index).to(device)).detach().cpu().numpy()
            auc = eval_roc_auc(label_tgt.detach().cpu().flatten(), pred.flatten())
            auc_list.append(auc)

        print(f"final auc: {auc}")
        print("==== DONE ====")
        
        if args.save:
            output = pd.DataFrame({
                "div": loss_div_list,
                "semantic": loss_semantic_list,
                "hetero": loss_hetero_list,
                "auc": auc_list
            })
            self.save(output, encoder, classifier)
        


    def save(self, df_log, encoder, classifier):
        args = self.args
        torch.save(encoder.state_dict(), os.path.join(args.savedir, f'encoder_{args.source}_to_{args.target}.pth'))
        torch.save(classifier.state_dict(), os.path.join(args.savedir, f'classifier_{args.source}_to_{args.target}.pth'))
        df_log.to_csv(os.path.join(args.outdir, f'log_{args.source}_to_{args.target}.csv'))
