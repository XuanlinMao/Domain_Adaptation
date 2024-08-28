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
            if printing: print(f"Epoch: {epoch}, Training loss: {loss_train}, Val loss: {loss_val}")
        
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
        savedir
    ):
        torch.save(self.source_encoder.state_dict(), os.path.join(savedir,f'{modulename}_encoder_{now()}.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(savedir,f'{modulename}_classifier_{now()}.pth'))
        self.get_perf_df().to_csv(os.path.join(outdir,f'{modulename}_perf_{now()}.csv'))
    
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

