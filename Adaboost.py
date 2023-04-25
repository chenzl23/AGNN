import torch
import torch.nn.functional as F
import math


def sample_weight_updating(weights, right_idx, err_idx, predictions, labels, pred_label, rho):
    n_samples = labels.shape[0]
    updating_rate_list = right_idx + err_idx
    for i in range(n_samples):
        label = pred_label[i]
        basic_updating_rate = predictions[i,label] / max(torch.sum(predictions[i,:]) - predictions[i,label], 1e-3)
        basic_updating_rate = torch.exp(torch.log(basic_updating_rate))
        if right_idx[i] == 1:
            updating_rate_list[i] = max(1.0 -  basic_updating_rate, rho)
        else:
            updating_rate_list[i] = 1.0 + basic_updating_rate
    return weights.mul(updating_rate_list)

def adaboosting_weight(Z_tp_list, Z_embed_list, labels, train_mask, num_class, rho, device):
    num_classifier = len(Z_tp_list)
    N = torch.sum(train_mask)
    classifier_weight_list = torch.zeros(num_classifier * 2)
    sample_weight_list = torch.ones(N).to(device) / N


    labels = labels[train_mask]
    
    for i in range(num_classifier):
        # Compute loss of Z_tp
        prediction_features = Z_tp_list[i]
        predictions = F.softmax(prediction_features, dim=1)
        pred_label = torch.argmax(predictions[train_mask], 1)
        err_idx = (pred_label != labels).float()
        right_idx = (pred_label == labels).float()
        
        sample_weight_list = sample_weight_updating(sample_weight_list, right_idx, err_idx, predictions[train_mask], labels, pred_label, rho)
        weighted_err = sample_weight_list.matmul(err_idx.t()) / torch.sum(sample_weight_list)
        classifier_weight_list[i] = 0.5 * torch.log(max(1 - weighted_err, 1e-4) / max(weighted_err, 1e-4)) + math.log(num_class - 1)

        # Compute loss of Z_embed
        prediction_features = Z_embed_list[i]
        predictions = F.softmax(prediction_features, dim=1)
        pred_label = torch.argmax(predictions[train_mask], 1)
        err_idx = (pred_label != labels).float()
        right_idx = (pred_label == labels).float()
        sample_weight_list = sample_weight_updating(sample_weight_list, right_idx, err_idx, predictions[train_mask], labels, pred_label, rho)
        weighted_err = sample_weight_list.matmul(err_idx.t()) / torch.sum(sample_weight_list)
        classifier_weight_list[num_classifier + i] = 0.5 * torch.log(max(1 - weighted_err, 1e-4) / max(weighted_err, 1e-4)) + math.log(num_class - 1)
        
    normalized_classifier_weight_list = torch.softmax(classifier_weight_list, dim=0)

    return normalized_classifier_weight_list

def compute_weighted_results(Z_tp_list, Z_embed_list, weights):
    result = Z_tp_list[0] * weights[0]
    num_classifier = weights.shape[0]
    num_layers = int(num_classifier / 2)
    for i in range(1, num_layers):
        result += Z_tp_list[i] * weights[i]
    for i in range(num_layers, num_classifier):
        result += Z_embed_list[i - num_layers] * weights[i]
    return result

