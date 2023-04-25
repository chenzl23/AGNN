import torch
import copy
import torch.nn.functional as F
from utils import accuracy
from Adaboost import adaboosting_weight, compute_weighted_results
import time 

def train(model, optimizer, dataloader, graph, args, device):
    best_valid_acc = 0.0
    patience = args.patience

    best_model = copy.deepcopy(model)
    best_model_weights = list()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = args.lr_patience, verbose = True, min_lr=1e-5)

    # Training
    for epoch in range(1, args.epoch_num + 1):
        tic = time.time()
        loss, valid_acc, weights = train_fullbatch(model, optimizer, scheduler, dataloader, args, device)

        
        if (valid_acc >= best_valid_acc):
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_model_weights = weights

        if args.early_stop:
            if (valid_acc >= best_valid_acc):
                patience = args.patience
            else:
                patience -= 1
                if (patience < 0):
                    print("Early Stopped!")
                    break

        toc = time.time()
        train_time = toc-tic
        
        print("Epoch: {0:d}".format(epoch), 
            "Training loss: {0:1.5f}".format(loss.cpu().detach().numpy()), 
            "Valid accuracy: {0:1.5f}".format(valid_acc),
            "Time used: {0:1.5f}".format(train_time)
            )

    test_model = best_model
    with torch.no_grad():
        test_model.eval()
        Z_tp_list, Z_embed_list = test_model(graph)
        weighted_prediction = compute_weighted_results(Z_tp_list, Z_embed_list, best_model_weights)
        predictions = F.log_softmax(weighted_prediction, dim=1)
        accuracy_value = accuracy(predictions[graph.test_mask], graph.y[graph.test_mask])
    print("Test accuracy: {0:1.5f}".format(accuracy_value))

    return accuracy_value


def train_fullbatch(model, optimizer, scheduler, graph, args, device):
    model.train()
    optimizer.zero_grad()
    Z_tp_list, Z_embed_list = model(graph)

    weights = adaboosting_weight(Z_tp_list, Z_embed_list, graph.y, graph.train_mask, args.num_class, args.rho, device)
    weighted_prediction = compute_weighted_results(Z_tp_list, Z_embed_list, weights)
    predictions = F.log_softmax(weighted_prediction, dim=1)
    loss = F.nll_loss(predictions[graph.train_mask], graph.y[graph.train_mask]) 
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # Evaluation Valid Set
    with torch.no_grad():
        model.eval()
        Z_tp_list, Z_embed_list = model(graph)
        weighted_prediction = compute_weighted_results(Z_tp_list, Z_embed_list, weights)
        predictions = F.log_softmax(weighted_prediction, dim=1)
        valid_acc = accuracy(predictions[graph.valid_mask], graph.y[graph.valid_mask])
    return loss, valid_acc, weights


