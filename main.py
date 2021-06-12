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

from datasets import TextLabelDataset, TextLabelDatasetBatch
from models import LabelEmbedModel, TextCNN, CombinedModel
from poincare_utils import PoincareDistance

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, label_embs):
        expanded_loss = self.loss(inputs, targets)
        _loss = torch.mean(expanded_loss,0)
        pt = torch.exp(-_loss)
        lam = ((1-pt)**self.gamma).detach()
        F_loss = lam* _loss
        return F_loss, expanded_loss

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
    def __init__(self, rho=1, use_geodesic=False, only_label=False):
        super(BiLevelLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.use_geodesic = use_geodesic
        self.scale_factor = 100
        self.rho = rho
        if use_geodesic or only_label:
            self.geo_loss = LabelLoss()
        self.only_label = only_label
        
    def forward(self, outputs, targets, label_embs):
        if self.only_label:
            return self.geo_loss(label_embs)
        expanded_loss = self.bce(outputs, targets)
        loss = torch.mean(expanded_loss,0)
        # loss = torch.pow(loss, self.rho)
        # if loss < 0:
        #     logging.error(outputs, targets)
        #     raise AssertionError
        if self.use_geodesic:
            loss1 = self.geo_loss(label_embs) / self.scale_factor
            return loss, loss1, expanded_loss
        return loss, expanded_loss

def train_epoch(combinedmodel, trainloader, criterion, optimizer, Y):
    losses = []
    for i, data in tqdm(enumerate(trainloader, 0)):
        docs, labels, edges = data
        docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
        optimizer.zero_grad()

        # doc_emb = doc_model(docs)
        # label_emb = label_model(Y)
        # dot = doc_emb @ label_emb.T
        # loss = criterion(dot, labels, label_model(edges))

        dot, label_edges = combinedmodel(docs, Y, edges)
        loss = criterion(dot, labels, label_edges)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    logging.info(f"\tTrain Loss {sum(losses)/len(losses):.6f}")


def eval(doc_model, label_model, dataloader, mode, Y, criterion):
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            docs, labels, edges = data
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()

            doc_emb = doc_model(docs)
            dot = doc_emb @ label_model(Y).T
            loss = criterion(dot, labels, label_model(edges))
            total_loss = loss.item()
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
    logging.info(f"\t{mode}: MicroF1-{micro_f.item():.4f}, MacroF1-{macro_f.item():.4f}. Loss-{total_loss}")
    return micro_f.item(), macro_f.item()

def eval2(combinedmodel, dataloader, mode, Y, criterion):
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            docs, labels, edges = data
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            dot, label_edges = combinedmodel(docs, Y, edges)
            loss = criterion(dot,labels,label_edges)
            total_loss = loss.item()
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
    logging.info(f"\t{mode}: MircoF1-{micro_f.item():.4f}, MacroF1-{macro_f.item():.4f}, Loss-{total_loss}")
    return micro_f.item(), macro_f.item(), (2 * macro_p * macro_r / (macro_p + macro_r + eps))

def eval_bilevel(combinedmodel, dataloader, mode, Y, weights, criterion, joint):
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    total_geodesic_loss = 0
    total_non_geodesic_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            docs, labels, edges = data
            docs = torch.squeeze(docs)
            labels = torch.squeeze(labels)
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            dot, label_edges = combinedmodel(docs, Y, edges)
            if joint:
                losses, geo_loss, _ = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights[:-1]) + weights[-1]*geo_loss
                total_loss += loss.item()
                total_geodesic_loss += geo_loss.item()
                total_non_geodesic_loss += losses.mean().item()
            else:
                losses, _ = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights)
                total_loss += loss.item()
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
    logging.info(f"\t{mode}: MircoF1-{micro_f.item():.4f}, MacroF1-{macro_f.item():.4f}, Loss-{total_loss}, Geodesic loss-{total_geodesic_loss}, Non Geodesic Loss-{total_non_geodesic_loss}")
    return micro_f.item(), macro_f.item(), (2 * macro_p * macro_r / (macro_p + macro_r + eps))

def train(
    combinedmodel, trainloader, valloader, testloader, criterion, optimizer, Y, epochs, save_folder
):
    best_macro = 0.0
    best_micro = 0.0
    bests = {"micro": (0, 0, 0), "macro": (0, 0, 0)}  # micro, macro, epoch
    test_f = []
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        combinedmodel.train()
        train_epoch(combinedmodel, trainloader, criterion, optimizer, Y)

        combinedmodel.eval()
        eval2(combinedmodel, trainloader, "Train", Y, criterion)
        micro_val, macro_val, _ = eval2(combinedmodel, valloader, "Val", Y, criterion)
        if epoch % 5 == 0:
            micro_f, macro_f, per_label_macro_f = eval2(combinedmodel, testloader, "Test", Y, criterion)
            test_f.append((micro_f, macro_f, epoch+1))
            if macro_val > best_macro:
                best_macro = macro_val
                bests["macro"] = (micro_val, macro_val, epoch + 1)
            if micro_val > best_micro:
                best_micro = micro_val
                bests["micro"] = (micro_val, macro_val, epoch + 1)
            with open("f1scores.txt",'a') as f:
                string = "\n".join(list(map(lambda x: str(x.item()),per_label_macro_f)))
                str_to_write = f"Epoch {epoch+1}/{epochs} \n" + string + "\n"
                f.write(str_to_write)
        # torch.save({
        #     'label_model': label_model.state_dict(),
        #     'doc_model': doc_model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'label_embs': label_model(Y),
        #     }, save_folder+'/'+str(epoch))
    best_test = {'micro': test_f[bests['micro'][2]-1], 'macro': test_f[bests['macro'][2]-1]}
    logging.info(best_test)

def train_bilevel(epochs, trainloader, valloader, testloader, combinedmodel, args_model_init, Y, optimizer, criterion, save_folder, wt_lr):
    freq = [10785, 673, 381, 947, 160, 4179, 49, 1172, 1462, 793, 189, 62, 922, 1058, 43, 443, 120, 312, 343, 2366, 1930, 399, 437, 285, 76, 246, 1205, 142, 202, 166, 196, 41, 31, 286, 0, 3448, 6970, 5881, 278, 679, 187, 65, 1255, 66, 449, 641, 15, 166, 94, 167, 17, 12, 8, 34, 407, 853, 43, 0, 3, 40, 102, 400, 54, 363, 1133, 233, 1004, 293, 106, 172, 6, 197, 471, 0, 13, 90, 1647, 166, 92, 37, 913, 23, 1115, 346, 135, 51, 49, 35, 59, 138, 38, 2, 45, 52, 2, 1293, 731, 1596, 2541, 943, 699, 1508, 311, 606]
    freq_t = torch.FloatTensor(freq)
    freq_t = freq_t/ freq_t.sum()
    freq_t = torch.pow(freq_t, args_model_init["rho"]).cuda()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    best_macro = 0.0
    best_micro = 0.0
    bests = {"micro": (0, 0, 0), "macro": (0, 0, 0)}  # micro, macro, epoch
    test_f = []
    if args_model_init["flat"]:
        weights = torch.ones(args_model_init["n_labels"]).cuda()
    else:
        weights = torch.ones(args_model_init["n_labels"]+1).cuda()
    for t in range(epochs):
        logging.info(f"Epoch {t+1}/{epochs}")
        print(f"Epoch {t+1}/{epochs}")
        total_loss = 0
        combinedmodel.train()
        for i,data in tqdm(enumerate(trainloader,0)):
            docs, labels, edges = data
            docs = torch.squeeze(docs)
            labels = torch.squeeze(labels)
            print(docs.shape, labels.shape)
            print(labels.sum(axis=0))
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            val_docs, val_labels, val_edges = next(iter(valloader))
            val_docs = torch.squeeze(val_docs)
            val_labels = torch.squeeze(val_labels)
            val_docs, val_labels, val_edges = val_docs.cuda(), val_labels.cuda(), val_edges.cuda()

            combinedmodel2 = CombinedModel(args_model_init)
            # combinedmodel2 = nn.DataParallel(combinedmodel2)
            combinedmodel2 = combinedmodel2.cuda()
            combinedmodel2.load_state_dict(copy.deepcopy(combinedmodel.state_dict()))
            optimizer2 = torch.optim.Adam(params=combinedmodel2.parameters(),lr=args_model_init["lr"])

            combinedmodel2.register_parameter('wts', torch.nn.Parameter(weights, requires_grad=True))

            with higher.innerloop_ctx(combinedmodel2, optimizer2) as (fmodel, fopt):
                dot, label_edges = fmodel(docs,Y,edges, freeze= True)
                if args_model_init["joint"]:
                    losses, geo_loss, _ = criterion(dot, labels, label_edges)
                    loss = torch.dot(losses, fmodel.wts[:-1]) + fmodel.wts[-1]*geo_loss
                    fopt.step(loss)
                else:
                    losses, exp = criterion(dot, labels, label_edges)
                    temp = torch.zeros(args_model_init["n_labels"]).cuda()
                    for d in range(args_model_init["n_labels"]):
                        mask = labels[:,d]==1
                        if mask.sum().item() == 0 :
                            temp[d] = 0
                            continue 
                        rem = exp[mask][:,d].mean()
                        temp[d] = rem
                    new_loss = temp*freq_t
                    loss = torch.dot(new_loss, fmodel.wts)
                    fopt.step(loss)
                val_dot, val_label_edges = fmodel(val_docs, Y, val_edges, freeze=True)
                if args_model_init["joint"]:
                    val_losses, geo_loss, val_exp = criterion(val_dot, val_labels, val_label_edges)
                else:
                    val_losses, val_exp = criterion(val_dot, val_labels, val_label_edges)
                temp = torch.tensor([0.0]).cuda()
                print_values = []
                for d in range(args_model_init["n_labels"]):
                    mask = val_labels[:,d]==1
                    if mask.sum().item() == 0 :
                        print_values.append(0)
                        continue 
                    rem = val_exp[mask][:,d].mean()
                    print_values.append(rem.item())
                    if temp.item() < rem.item():
                        index = d
                    temp = torch.max(temp, rem)
                # temp = torch.max(val_losses)
                # index = torch.argmax(val_losses).item()
                # if args_model_init["joint"]:
                #     if temp.item() < geo_loss.item():
                #         index = args_model_init['n_labels']
                #     temp = torch.max(temp, geo_loss)
                #     # print(temp.item())
                # if args_model_init["joint"]:
                #     if temp.item() < geo_loss.item():
                #         index = args_model_init['n_labels']
                #     temp = torch.max(temp, geo_loss)
                val_i_file = os.path.join(save_folder,'val_index.txt')
                with open(val_i_file , 'a') as f:
                    string = str(t) + ","+str(i)+","+ str(index) +","+ str(temp.item())+"," +"\n" 
                    f.write(string)
                val_l_file = os.path.join(save_folder,'val_losses.txt')
                with open(val_l_file,'a') as f:
                    string = str(t) + "," + str(i) + "," + ",".join(list(map(str, print_values))) + "\n"
                    # string = str(t) + "," + str(i) + "," + ",".join(list(map(str, list(val_losses.detach().cpu().numpy())))) + "\n"
                    f.write(string)
                wt_grads = torch.autograd.grad(temp, fmodel.parameters(time=0), allow_unused=True)[0]
            weights = weights - wt_lr * wt_grads
            weights = torch.clamp(weights, min=0)
            del wt_grads
            optimizer.zero_grad()
            dot, label_edges = combinedmodel(docs, Y, edges)
            # labels_pop = labels.sum(axis=0)/(1.0*labels.sum())
            if args_model_init["joint"]:
                losses, geo_loss, _ = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights[:-1]) + weights[-1]*geo_loss
            else:
                losses, exp = criterion(dot, labels, label_edges)
                # multiplicand = 1 + combinedmodel.C*torch.exp(-combinedmodel.A*torch.log(combinedmodel.B + labels_pop))
                # losses = multiplicand*losses
                temp = torch.zeros(args_model_init["n_labels"]).cuda()
                for d in range(args_model_init["n_labels"]):
                    mask = labels[:,d]==1
                    if mask.sum().item() == 0 :
                        temp[d] = 0
                        continue 
                    rem = exp[mask][:,d].mean()
                    temp[d] = rem
                new_loss = temp*freq_t
                loss = torch.dot(new_loss, weights)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        # logging.info(f"Total training loss: {total_loss}")
        combinedmodel.eval()
        eval_bilevel(combinedmodel, trainloader, "Train", Y, weights, criterion, args_model_init["joint"])
        micro_val, macro_val, _ = eval_bilevel(combinedmodel, valloader, "Val", Y, weights, criterion, args_model_init["joint"])
        if t % 5 == 0:
            micro_f, macro_f, per_label_macro_f = eval_bilevel(combinedmodel, testloader, "Test", Y, weights, criterion, args_model_init["joint"])
            test_f.append((micro_f, macro_f, t+1))
            if macro_val > best_macro:
                best_macro = macro_val
                bests["macro"] = (micro_val, macro_val, t + 1)
            if micro_val > best_micro:
                best_micro = micro_val
                bests["micro"] = (micro_val, macro_val, t + 1)
            f1_file = os.path.join(save_folder,"f1scores.txt")
            with open(f1_file,'a') as f:
                string = ",".join(list(map(lambda x: str(x.item()),per_label_macro_f)))
                str_to_write = f"Epoch {t+1}/{epochs} \n" + string + "\n"
                f.write(str_to_write)
        print(f"Total loss: {total_loss}")
        weight_file = os.path.join(save_folder,"weights.txt")
        with open(weight_file,'a') as f:
            string = ", ".join(list(map(lambda x: str(x.item()),weights)))
            str_to_write = f"Epoch {t+1}/{epochs} \n" + string + "\n"
            f.write(str_to_write)
        # torch.save({
        #     'combinedmodel': combinedmodel.state_dict(),
        #     'optimizer': optimizer.state_dict()
        #     }, save_folder+'/'+str(t))
    best_test = {'micro': test_f[bests['micro'][2]-1], 'macro': test_f[bests['macro'][2]-1]}
    logging.info(best_test)
    return weights

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
    parser.add_argument('--rho',default = 1,type=float)
    parser.add_argument('--gamma', default=0, type=float)
    args = parser.parse_args()

    os.makedirs(args.exp_name, exist_ok=True)

    logging.basicConfig(filename=args.exp_name+'/res.txt', level=logging.DEBUG)
    logging.info(args)

    # Datasets and Dataloaders
    try:
        trainvalset = pickle.load(open(f"{args.dataset}/train.pkl", "rb"))
    except:
        # json_data_file, label_file, vocab_dict=None, n_tokens=256, nnegs=5
        trainvalset = TextLabelDatasetBatch(f"{args.dataset}/{args.dataset}_train.json", f"{args.dataset}/{args.dataset}_labels.txt", None, 256, 5)
        pickle.dump(trainvalset, open(f"{args.dataset}/train.pkl", "wb"))

    # Split into train and val sets
    trainset, valset = torch.utils.data.dataset.random_split(trainvalset, 
                [int(0.9*len(trainvalset)), len(trainvalset)- int(0.9*len(trainvalset))])

    if args.dataset=='yelp':
        trainloader = DataLoader(
            trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )
    else:
        trainloader = DataLoader(
            trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )

    valloader = DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    try:
        testset = pickle.load(open(f"{args.dataset}/test.pkl", "rb"))
    except:
        testset = TextLabelDataset(f"{args.dataset}/{args.dataset}_test.json", f"{args.dataset}/{args.dataset}_labels.txt", trainvalset.text_dataset.vocab, 256)
        pickle.dump(testset, open(f"{args.dataset}/test.pkl", "wb"))

    testloader = DataLoader(
        testset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )


    glove_file = "GloVe/glove.6B.300d.txt"
    if not args.flat:
        emb_dim = 300  # Document and label embed length
    else:
        emb_dim = trainvalset.n_labels
    word_embed_dim = 300


    # Model
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
        rho = 1, use_geodesic=args.joint, only_label=args.cascaded_step1
    )
    # criterion = FocalLoss(gamma=args.gamma)
    # optimizer = torch.optim.Adam([
    #     {'params': doc_model.parameters(), 'lr': doc_lr},
    #     {'params': label_model.parameters(), 'lr': 0.001}
    # ])


    # logging.info('Starting Training')
    # # Train and evaluate
    Y = torch.arange(trainvalset.n_labels).cuda()

    args_model_init = {
        "n_labels":trainvalset.n_labels,
        "lr" : doc_lr,
        "vocab" : trainvalset.text_dataset.vocab,
        "glove_file" : glove_file,
        "emb_dim" : emb_dim,
        "drop_p_doc" : 0.1,
        "word_embed_dim" : word_embed_dim,
        "drop_p_label" : 0.6,
        "flat" : args.flat,
        "joint" : args.joint,
        "rho" : args.rho
    }
    combinedmodel = CombinedModel(args_model_init)
    # combinedmodel = nn.DataParallel(combinedmodel)
    combinedmodel = combinedmodel.cuda()
    optimizer = torch.optim.Adam(params=combinedmodel.parameters(),lr=args_model_init["lr"])
    # train(
    #     combinedmodel,
    #     trainloader,
    #     valloader,
    #     testloader,
    #     criterion,
    #     optimizer,
    #     Y,
    #     args.num_epochs,
    #     args.exp_name
    # )

    train_bilevel(
        args.num_epochs,
        trainloader,
        valloader,
        testloader,
        combinedmodel,
        args_model_init,
        Y,
        optimizer,
        criterion,
        save_folder='checkpoints_pop_rho_'+str(args.rho),
        wt_lr= 0.1
    )
