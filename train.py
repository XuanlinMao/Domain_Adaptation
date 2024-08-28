
import torch
import torch.optim as optim
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np


def train(
        n_epoch,
        lr,
        data_src,
        label_src,
        source_encoder,
        classifier,
        device,
        early_stopping:bool=True,
        n_stopping:int=20
):
    loss_val_best = np.inf
    stops = 0

    source_encoder.reset_parameters()
    classifier.reset_parameters()
    source_encoder = source_encoder.to(device)
    classifier = classifier.to(device)

    source_optimizer = optim.Adam([{'params':source_encoder.parameters(),'lr':lr},
                                {'params':classifier.parameters(),'lr':lr}])
    loss_bce = nn.BCELoss()

    print("== Start Training ===")
    for epoch in range(n_epoch):
        source_encoder.train()
        classifier.train()

        source_encodings = source_encoder(data_src.x,data_src.edge_index)
        source_pred = classifier(source_encodings)
        loss = loss_bce(source_pred[data_src.train_mask,:], label_src[data_src.train_mask,:])
        source_optimizer.zero_grad()
        loss.backward()
        loss_train = loss.detach().cpu().numpy()
        source_optimizer.step()

        with torch.no_grad():
            source_encoder.eval()
            classifier.eval()
            loss_val = loss_bce(source_pred[data_src.val_mask,:], label_src[data_src.val_mask,:]).detach().cpu().numpy()
        
        torch.cuda.empty_cache()
        if early_stopping:
            if loss_val < loss_val_best:
                stops = 0
                loss_val_best = loss_val
                source_encoder_best = deepcopy(source_encoder)
                classifier_best = deepcopy(classifier)
                epoch_best = epoch
            else:
                stops += 1
            if stops >= n_stopping:
                print(f"=== Early Stopping at epoch {epoch_best}, best loss_val = {loss_val_best} ===")
                break
        print(f"Epoch: {epoch}, Training loss: {loss_train}, Val loss: {loss_val}")
    
    print("=== DONE ===")
    if early_stopping:
        return source_encoder_best, classifier_best
    else:
        return source_encoder, classifier
    

def validate(
        data_src,
        label_src,
        source_encoder,
        classifier
    ):
    loss_bce = nn.BCELoss()
    source_encodings = source_encoder(data_src.x,data_src.edge_index)
    source_pred = classifier(source_encodings)
    with torch.no_grad():
        source_encoder.eval()
        classifier.eval()
        loss_val = loss_bce(source_pred[data_src.val_mask,:], label_src[data_src.val_mask,:]).detach().cpu().numpy()
    return loss_val