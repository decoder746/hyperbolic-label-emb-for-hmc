from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import logging
import numpy as np
import copy
import higher

from datasets import TextLabelDataset
from models import LabelEmbedModel, TextCNN, CombinedModel
from poincare_utils import PoincareDistance

class LabelLoss(nn.Module):
    def __init__(self, dist=PoincareDistance):
        super(LabelLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.dist = dist

    def forward(self, e):
        # Project to Poincare Ball
        e = e/(1+torch.sqrt(1+e.norm(dim=-1, keepdim=True)**2))
        # Within a batch take the embeddings of all but the first component
        o = e.narrow(1, 1, e.size(1) - 1)
        # Embedding of the first component
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist.apply(s, o).squeeze(-1)
        # Distance between the first component and all the remaining component (embeddings of)
        outputs = -dists
        targets = torch.zeros(outputs.shape[0]).long().cuda()
        return self.loss(outputs, targets)


class Loss(nn.Module):
    def __init__(self, use_geodesic=False, _lambda=None, only_label=False):
        super(Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_geodesic = use_geodesic
        self._lambda = _lambda
        if use_geodesic or only_label:
            self.geo_loss = LabelLoss()
        self.only_label = only_label

    def forward(self, outputs, targets, label_embs):
        if self.only_label:
            return self.geo_loss(label_embs)

        loss = self.bce(outputs, targets)
        # if loss < 0:
        #     logging.error(outputs, targets)
        #     raise AssertionError
        if self.use_geodesic:
            loss1 = self.geo_loss(label_embs)
            loss += self._lambda * loss1
        return loss

class BiLevelLoss(nn.Module):
    def __init__(self, use_geodesic=False, only_label=False):
        super(BiLevelLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.use_geodesic = use_geodesic
        if use_geodesic or only_label:
            self.geo_loss = LabelLoss()
        self.only_label = only_label
        
    def forward(self, outputs, targets, label_embs):
        if self.only_label:
            return self.geo_loss(label_embs)
        loss = torch.sum(self.bce(outputs, targets),0)
        if loss < 0:
            logging.error(outputs, targets)
            raise AssertionError
        if self.use_geodesic:
            loss1 = self.geo_loss(label_embs)
            loss = torch.cat((loss,loss1))
        return loss

def train_epoch(doc_model, label_model, trainloader, criterion, optimizer, Y):
    losses = []
    for i, data in tqdm(enumerate(trainloader, 0)):
        docs, labels, edges = data
        docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
        optimizer.zero_grad()

        doc_emb = doc_model(docs)
        label_emb = label_model(Y)
        dot = doc_emb @ label_emb.T
        loss = criterion(dot, labels, label_model(edges))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    logging.info(f"\tTrain Loss {sum(losses)/len(losses):.6f}")


def eval(doc_model, label_model, dataloader, mode, Y):
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            docs, labels, _ = data
            docs, labels = docs.cuda(), labels.cuda()

            doc_emb = doc_model(docs)
            dot = doc_emb @ label_model(Y).T
            t = torch.sigmoid(dot)

            y_pred = 1.0 * (t > 0.5)
            y_true = labels

            tp += (y_true * y_pred).sum(dim=0)
            fp += ((1 - y_true) * y_pred).sum(dim=0)
            fn += (y_true * (1 - y_pred)).sum(dim=0)

    eps = 1e-7
    p = tp.sum() / (tp.sum() + fp.sum() + eps)
    r = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f = 2 * p * r / (p + r + eps)
    macro_p = tp / (tp + fp + eps)
    macro_r = tp / (tp + fn + eps)
    macro_f = (2 * macro_p * macro_r / (macro_p + macro_r + eps)).mean()
    logging.info(f"\t{mode}: {micro_f.item():.4f}, {macro_f.item():.4f}")
    return micro_f.item(), macro_f.item()


def train(
    doc_model, label_model, trainloader, valloader, testloader, criterion, optimizer, Y, epochs, save_folder
):
    best_macro = 0.0
    best_micro = 0.0
    bests = {"micro": (0, 0, 0), "macro": (0, 0, 0)}  # micro, macro, epoch
    test_f = []
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        label_model = label_model.train()
        doc_model = doc_model.train()
        train_epoch(doc_model, label_model, trainloader, criterion, optimizer, Y)

        label_model = label_model.eval()
        doc_model = doc_model.eval()
        eval(doc_model, label_model, trainloader, "Train", Y)
        micro_val, macro_val = eval(doc_model, label_model, valloader, "Val", Y)
        micro_f, macro_f = eval(doc_model, label_model, testloader, "Test", Y)
        test_f.append((micro_f, macro_f, epoch+1))
        if macro_val > best_macro:
            best_macro = macro_val
            bests["macro"] = (micro_val, macro_val, epoch + 1)
        if micro_val > best_micro:
            best_micro = micro_val
            bests["micro"] = (micro_val, macro_val, epoch + 1)

        torch.save({
            'label_model': label_model.state_dict(),
            'doc_model': doc_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'label_embs': label_model(Y),
            }, save_folder+'/'+str(epoch))
    best_test = {'micro': test_f[bests['micro'][2]-1], 'macro': test_f[bests['macro'][2]-1]}
    logging.info(best_test)

def train_bilevel(epochs, trainloader, valloader, combinedmodel, args_model_init, Y, optimizer, wt_lr):
    weights = torch.ones(args_model_init["n_labels"]).cuda()
    for t in range(epochs):
        for i,data in tqdm(enumerate(trainloader,0)):
            docs, labels, edges = data
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            val_docs, val_labels, val_edges = next(iter(valloader))
            val_docs, val_labels, val_edges = val_docs.cuda(), val_labels.cuda(), val_edges.cuda()

            combinedmodel2 = CombinedModel(args_model_init)
            combinedmodel2 = nn.DataParallel(combinedmodel2)
            combinedmodel2 = combinedmodel2.cuda()
            combinedmodel2.load_state_dict(copy.deepcopy(combinedmodel.state_dict()))
            optimizer2 = torch.optim.Adam(params=combinedmodel2.parameters(),lr=args_model_init["lr"])

            combinedmodel2.register_parameter('wts', torch.nn.Parameter(weights, requires_grad=True))

            with higher.innerloop_ctx(combinedmodel2, optimizer2) as (fmodel, fopt):
                doc_emb, label_emb, label_edges = fmodel(docs,Y,edges)
                dot = doc_emb @ label_emb.T
                losses = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, fmodel.wts)
                fopt.step(loss)

                val_doc_emb, val_label_emb, val_label_edges = fmodel(val_docs, Y, val_edges)
                val_dot = val_doc_emb @ val_label_emb.T
                val_losses = criterion(val_dot, val_labels, val_label_edges)
                temp = torch.tensor([0]).cuda()
                for d in range(args_model_init["n_labels"]):
                    temp = torch.maximum(temp, val_losses[torch.where(Y==d)].sum())
                wt_grads = torch.autograd.grad(temp, fmodel.parameters(time=0))[0]
            weights = weights - wt_lr * wt_grads
            weights = torch.clamp(weights, min=0)
            optimizer.zero_grad()
            doc_emb, label_emb, label_edges = combinedmodel(docs, Y, edges)
            dot = doc_emb @ label_emb.T
            losses = criterion(dot, labels, label_edges)
            loss = torch.dot(losses, weights)
            loss.backward()
            optimizer.step()
        if (t+1)%1 == 0:
            print(weights)
            print(f"EPOCH: {t+1}, train_loss: {loss.item()}")
            print(" ")
        if (t+1)% 5 == 0:
            print("saving model")
            torch.save(combinedmodel.state_dict(), f"combinedModelEpoch{t+1}.pt")
    return weights

# def get_new_weights(lambda_list, doc_model, label_model, trainloader, valloader, criterion, optimizer, e1, e2):
#     L_hat_train = 0
#     Ui_loss_params_list = []
#     for i in range(len(lambda_list)):
#         for param in doc_model.parameters():
#             param.grad.data.zero_()
#         for param in label_model.parameters():
#             param.grad.data.zero_()
#         L_Ui = 0
#         trainloader2 = copy.deepcopy(trainloader)
#         for j, data in tqdm(enumerate(trainloader2, 0)):
#             docs, labels, _ = data
#             docs, labels = docs.cuda(), labels.cuda()
#             doc_emb = doc_model(docs)
#             label_emb = label_model(Y)
#             dot = doc_emb @ label_emb.T
#             L_Ui += nn.BCELoss(dot[i],labels[i])
#         grads_list_Ui = []
#         for param in doc_model.parameters():
#             grads_list_Ui.append(param.grad.view(-1))

#         Ui_loss_params = torch.cat(grads_list_Ui)
#         Ui_loss_params_list.append(Ui_loss_params)

#         L_hat_train = L_hat_train + lambda_list[i]*L_Ui
#     optimizer.zero_grad()
#     L_hat_train.backward()
#     optimizer.step()

#     val_loss_list = []
#     for i in range(len(lambda_list)):
#         L_vi = 0
#         valloader2 = copy.deepcopy(valloader)
#         for j, data in tqdm(enumerate(valloader2, 0)):
#             docs, labels, _ = data
#             docs, labels = docs.cuda(), labels.cuda()
#             doc_emb = doc_model(docs)
#             label_emb = label_model(Y)
#             dot = doc_emb @ label_emb.T
#             L_vi += nn.BCELoss(dot[i],labels[i])
#         val_loss_list.append(L_vi)
#     i_t = np.argmax(np.array(val_loss_list))
#     for param in doc_model.parameters():
#         param.grad.data.zero_()
#     val_loss_list[i_t].backward()
#     grads_list_val = []
#     for param in doc_model.parameters():
#         grads_list_val.append(param.grad.view(-1))
#     val_loss_params = torch.cat(grads_list_val)
#     for i in range(len(lambda_list)):
#         lambda_list[i] = lambda_list[i] + e1*e2*torch.dot(val_loss_params, Ui_loss_params_list[i])
#     if lambda_list[i] < 0:
#         lambda_list[i] = 0
#     return lambda_list

# def train_bilevel(doc_model, label_model, trainloader, valloader, testloader, criterion, optimizer, Y, epochs, n_labels,e1,e2):
#     lambda_list = [1 for i in range(n_labels)]
#     for epoch in range(epochs):
#         logging.info(f"Epoch {epoch+1}/{epochs}")
#         label_model = label_model.train()
#         doc_model = doc_model.train()
#         lambda_list = get_new_weights(lambda_list,doc_model,label_model,trainloader,valloader,criterion,optimizer,e1,e2)
#         train_loss = 0
#         for i in range(len(lambda_list)):
#             L_Ui = 0
#             trainloader2 = copy.deepcopy(trainloader)
#             for j, data in tqdm(enumerate(trainloader2, 0)):
#                 docs, labels, _ = data
#                 docs, labels = docs.cuda(), labels.cuda()
#                 doc_emb = doc_model(docs)
#                 label_emb = label_model(Y)
#                 dot = doc_emb @ label_emb.T
#                 L_Ui += nn.BCELoss(dot[i],label_emb[i])
#             train_loss = train_loss + lambda_list[i] * L_Ui
#         if epoch % 1 == 0:
#             print(train_loss.item())
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--flat', default=False, action='store_true')
    parser.add_argument('--cascaded_step1', default=False, action='store_true')
    parser.add_argument('--cascaded_step2', default=False, action='store_true')
    parser.add_argument('--joint', default=False, action='store_true')
    parser.add_argument('--pretrained_label_model', default=None)
    parser.add_argument('--dataset', default='rcv1', choices=['rcv1', 'nyt', 'yelp'])
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--geodesic_lambda', default=0.1, type=float)
    args = parser.parse_args()

    os.makedirs(args.exp_name, exist_ok=True)

    logging.basicConfig(filename=args.exp_name+'/res.txt', level=logging.DEBUG)
    logging.info(args)

    # Datasets and Dataloaders
    try:
        trainvalset = pickle.load(open(f"{args.dataset}/train.pkl", "rb"))
    except:
        # json_data_file, label_file, vocab_dict=None, n_tokens=256, nnegs=5
        trainvalset = TextLabelDataset(f"{args.dataset}/{args.dataset}_train.json", f"{args.dataset}/{args.dataset}_labels.txt", None, 256, 5)
        pickle.dump(trainvalset, open(f"{args.dataset}/train.pkl", "wb"))

    # Split into train and val sets
    trainset, valset = torch.utils.data.dataset.random_split(trainvalset, 
                [int(0.9*len(trainvalset)), len(trainvalset)- int(0.9*len(trainvalset))])

    if args.dataset=='yelp':
        trainloader = DataLoader(
            trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
        )
    else:
        trainloader = DataLoader(
            trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

    valloader = DataLoader(
        valset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    # try:
    #     testset = pickle.load(open(f"{args.dataset}/test.pkl", "rb"))
    # except:
    #     testset = TextLabelDataset(f"{args.dataset}/{args.dataset}_test.json", f"{args.dataset}/{args.dataset}_labels.txt", trainvalset.text_dataset.vocab, 256)
    #     pickle.dump(testset, open(f"{args.dataset}/test.pkl", "wb"))

    # testloader = DataLoader(
    #     testset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True
    # )


    glove_file = "GloVe/glove.6B.300d.txt"
    if not args.flat:
        emb_dim = 300  # Document and label embed length
    else:
        emb_dim = trainvalset.n_labels
    word_embed_dim = 300


    # # Model
    # doc_model = TextCNN(
    #     trainvalset.text_dataset.vocab,
    #     glove_file=glove_file,
    #     emb_dim=emb_dim,
    #     dropout_p=0.1,
    #     word_embed_dim=word_embed_dim,
    # )
    doc_lr = 0.001
    # label_model = LabelEmbedModel(trainvalset.n_labels, emb_dim=emb_dim, dropout_p=0.6, eye=args.flat)

    # if args.cascaded_step2:
    #     label_model_pretrained = torch.load(args.pretrained_label_model)['label_model']
    #     label_model.load_state_dict(label_model_pretrained)

    # if args.flat or args.cascaded_step2:
    #     for param in label_model.parameters():
    #         param.require_grad = False

    # doc_model = nn.DataParallel(doc_model)
    # label_model = nn.DataParallel(label_model)

    # doc_model = doc_model.cuda()
    # label_model = label_model.cuda()

    # # Loss and optimizer
    # criterion = Loss(
    #     use_geodesic=args.joint, _lambda=args.geodesic_lambda, only_label=args.cascaded_step1
    # )

    criterion = BiLevelLoss(
        use_geodesic=args.joint, only_label=args.cascaded_step1
    )
    # optimizer = torch.optim.Adam([
    #     {'params': doc_model.parameters(), 'lr': doc_lr},
    #     {'params': label_model.parameters(), 'lr': 0.001}
    # ])


    # logging.info('Starting Training')
    # # Train and evaluate
    Y = torch.arange(trainvalset.n_labels).cuda()
    # train(
    #     doc_model,
    #     label_model,
    #     trainloader,
    #     valloader,
    #     testloader,
    #     criterion,
    #     optimizer,
    #     Y,
    #     args.num_epochs,
    #     args.exp_name
    # )

    args_model_init = {
        "n_labels":trainvalset.n_labels,
        "lr" : doc_lr,
        "vocab" : trainvalset.text_dataset.vocab,
        "glove_file" : glove_file,
        "emb_dim" : emb_dim,
        "drop_p_doc" : 0.1,
        "word_embed_dim" : word_embed_dim,
        "drop_p_label" : 0.6,
        "flat" : args.flat
    }
    combinedmodel = CombinedModel(args_model_init)
    combinedmodel = nn.DataParallel(combinedmodel)
    combinedmodel = combinedmodel.cuda()
    optimizer = torch.optim.Adam(params=combinedmodel.parameters(),lr=args_model_init["lr"])
    train_bilevel(
        args.num_epochs,
        trainloader,
        valloader,
        combinedmodel,
        args_model_init,
        Y,
        optimizer,
        wt_lr= 0.001
    )
    # train_bilevel(
    #     doc_model,
    #     label_model,
    #     trainloader,
    #     valloader,
    #     testloader,
    #     criterion,
    #     optimizer,
    #     Y,
    #     args.num_epochs,
    #     trainvalset.n_labels,
    #     e1=0.01,
    #     e2=0.01
    # )
