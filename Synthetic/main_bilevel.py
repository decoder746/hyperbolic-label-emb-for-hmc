from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import logging
import numpy as np
import copy
import higher
import random

from poincare_utils import PoincareDistance

class LabelDataset(Dataset):
    def __init__(self, N, nnegs=5):
        super(LabelDataset, self).__init__()
        self.items = N.shape[0]
        self.len = N.shape[0]*N.shape[0]
        self.N = N
        self.nnegs = nnegs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration
        t = idx//self.items
        h = idx%self.items
        negs = np.arange(self.items)[self.N[t][h] == 1.0]
        negs = negs.repeat(self.nnegs)
        np.random.shuffle(negs)
        return torch.tensor([t, h, *negs[:self.nnegs]])

class Synthetic(Dataset):
    def __init__(self, drop_prob, size):
        side = 4
        n = side//2
        self.means = [(x,y) for x in range(-n,n) for y in range(-n,n)]
        self.covs = [0.01 for x in range(-n,n) for y in range(-n,n)]
        self.N = np.zeros((21,21))
        self.x = []
        self.y = []
        self.size = size
        self.drop_prob = drop_prob
        self.per_label = dict(zip(list(range(21)),[0]*21))
        for i in range(self.size):
            x, y = self.getitem(i)
            for ele in y:
                self.per_label[ele] += 1
            self.x.append(x)
            self.y.append(y)

        print(self.per_label)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.x[i], self.y[i]


    @staticmethod
    def sample(mean, cov):
        return np.random.multivariate_normal(mean, cov)

    def multihot(self, x):
        with torch.no_grad():
            ret = torch.zeros(21)
            for l in x:
                ret += nn.functional.one_hot(torch.tensor(l), 21)
        return ret

    def getitem(self, idx):
        i = np.random.randint(len(self.means))
        x = Synthetic.sample(self.means[i], self.covs[i]*np.eye(2))
        y = [i, 20]
        if i==0:
            a = random.uniform(0,1)
            if a > 0.5:
                y.append(1)
        if i in [0,1,4,5]:
            y.append(16)
        elif i in [2,3,6,7]:
            y.append(17)
        elif i in [8,9,12,13]:
            y.append(18)
        elif i in [10,11,14,15]:
            y.append(19)

        # Missing Labels here
        if np.random.rand()>1.0-self.drop_prob:
            np.random.shuffle(y)
            y = y[:2]
        for i in y:
            for j in y:
                self.N[i,j]+=1
                self.N[j,i]+=1
        y = self.multihot(y)
        return torch.tensor(x, dtype=torch.float), y


class TextLabelDataset(Dataset):
    def __init__(self, drop_prob, size):
        super(TextLabelDataset, self).__init__()

        self.text_dataset = Synthetic(drop_prob, size)
        
        similarity_matrix = torch.tensor(self.text_dataset.N)
        n_labels = 21
        N = torch.zeros((n_labels, n_labels, n_labels))
        for i in range(n_labels):
            for j in range(n_labels):
                N[i][j] = 1.0*(similarity_matrix[i, :] <= similarity_matrix[i][j])

        self.label_dataset = LabelDataset(N)

        self.n_labels = n_labels
        self.len = max(len(self.text_dataset), len(self.label_dataset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return *self.text_dataset[idx%len(self.text_dataset)], self.label_dataset[idx%len(self.label_dataset)]



class PointModel(nn.Module):
    def __init__(self, n_labels, emb_dim):
        super(PointModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, emb_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class LabelEmbedModel(nn.Module):
    def __init__(self, n_labels, emb_dim, eye=False):
        super(LabelEmbedModel, self).__init__()
        self.eye = eye
        self.e = nn.Embedding(
                    n_labels, emb_dim,
                    max_norm=1.0,
                    scale_grad_by_freq=False
                )
        self.init_weights()

    def init_weights(self, scale=1e-4):
        if self.eye:
            torch.nn.init.eye_(self.e.weight)
        else:
            self.e.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, x):
        return self.e(x)

class CombinedModel(nn.Module):
    def __init__(self, args_model_init):
        super(CombinedModel, self).__init__()
        self.doc_model = PointModel(args_model_init["n_labels"],args_model_init["emb_dim"])
        self.label_model = LabelEmbedModel(args_model_init["n_labels"], emb_dim=args_model_init["emb_dim"], eye=args_model_init["flat"])
        if args_model_init["flat"]:
            for param in self.label_model.parameters():
                param.require_grad = False

    def forward(self, x, y, z):
        return self.doc_model(x), self.label_model(y), self.label_model(z)

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
        self.scale_factor = 1
        if use_geodesic or only_label:
            self.geo_loss = LabelLoss()
        self.only_label = only_label
        
    def forward(self, outputs, targets, label_embs):
        if self.only_label:
            return self.geo_loss(label_embs)
        loss = torch.mean(self.bce(outputs, targets),0)
        # if loss < 0:
        #     logging.error(outputs, targets)
        #     raise AssertionError
        if self.use_geodesic:
            loss1 = self.geo_loss(label_embs) / self.scale_factor
            return loss, loss1
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

def eval_bilevel(combinedmodel, dataloader, mode, Y, weights, criterion, joint):
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    total_geodesic_loss = 0
    total_non_geodesic_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            docs, labels, edges = data
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            doc_emb, label_emb, label_edges = combinedmodel(docs, Y, edges)
            dot = doc_emb @ label_emb.T
            if joint:
                losses, geo_loss = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights[:-1]) + weights[-1]*geo_loss
                total_loss += loss.item()
                total_geodesic_loss += geo_loss.item()
                total_non_geodesic_loss += losses.mean().item()
            else:
                losses = criterion(dot, labels, label_edges)
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
        eval(doc_model, label_model, trainloader, "Train", Y, criterion)
        micro_val, macro_val = eval(doc_model, label_model, valloader, "Val", Y, criterion)
        micro_f, macro_f = eval(doc_model, label_model, testloader, "Test", Y, criterion)
        test_f.append((micro_f, macro_f, epoch+1))
        if macro_val > best_macro:
            best_macro = macro_val
            bests["macro"] = (micro_val, macro_val, epoch + 1)
        if micro_val > best_micro:
            best_micro = micro_val
            bests["micro"] = (micro_val, macro_val, epoch + 1)

        # torch.save({
        #     'label_model': label_model.state_dict(),
        #     'doc_model': doc_model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'label_embs': label_model(Y),
        #     }, save_folder+'/'+str(epoch))
    best_test = {'micro': test_f[bests['micro'][2]-1], 'macro': test_f[bests['macro'][2]-1]}
    logging.info(best_test)

def train_bilevel(epochs, trainloader, valloader, testloader, combinedmodel, args_model_init, Y, optimizer, criterion, exp_name, save_folder, wt_lr):
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
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            val_docs, val_labels, val_edges = next(iter(valloader))
            val_docs, val_labels, val_edges = val_docs.cuda(), val_labels.cuda(), val_edges.cuda()

            combinedmodel2 = CombinedModel(args_model_init)
            # combinedmodel2 = nn.DataParallel(combinedmodel2)
            combinedmodel2 = combinedmodel2.cuda()
            combinedmodel2.load_state_dict(copy.deepcopy(combinedmodel.state_dict()))
            optimizer2 = torch.optim.Adam(params=combinedmodel2.parameters(),lr=args_model_init["lr"])

            combinedmodel2.register_parameter('wts', torch.nn.Parameter(weights, requires_grad=True))

            with higher.innerloop_ctx(combinedmodel2, optimizer2) as (fmodel, fopt):
                doc_emb, label_emb, label_edges = fmodel(docs,Y,edges)
                dot = doc_emb @ label_emb.T
                if args_model_init["joint"]:
                    losses, geo_loss = criterion(dot, labels, label_edges)
                    loss = torch.dot(losses, fmodel.wts[:-1]) + fmodel.wts[-1]*geo_loss
                    fopt.step(loss)
                else:
                    losses = criterion(dot, labels, label_edges)
                    loss = torch.dot(losses, fmodel.wts)
                    fopt.step(loss)
                val_doc_emb, val_label_emb, val_label_edges = fmodel(val_docs, Y, val_edges)
                val_dot = val_doc_emb @ val_label_emb.T
                if args_model_init["joint"]:
                    val_losses, geo_loss = criterion(val_dot, val_labels, val_label_edges)
                else:
                    val_losses = criterion(val_dot, val_labels, val_label_edges)
                temp = torch.tensor([0.0]).cuda()
                for d in range(args_model_init["n_labels"]):
                    temp = torch.max(temp, val_losses[torch.where(Y==d)].sum())
                if args_model_init["joint"]:
                    temp = torch.max(temp, geo_loss)
                wt_grads = torch.autograd.grad(temp, fmodel.parameters(time=0))[0]
            weights = weights - wt_lr * wt_grads
            weights = torch.clamp(weights, min=0)
            optimizer.zero_grad()
            doc_emb, label_emb, label_edges = combinedmodel(docs, Y, edges)
            dot = doc_emb @ label_emb.T
            if args_model_init["joint"]:
                losses, geo_loss = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights[:-1]) + weights[-1]*geo_loss
            else:
                losses = criterion(dot, labels, label_edges)
                loss = torch.dot(losses, weights)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        # logging.info(f"Total training loss: {total_loss}")
        combinedmodel.eval()
        eval_bilevel(combinedmodel, trainloader, "Train", Y, weights, criterion, args_model_init["joint"])
        micro_val, macro_val, _ = eval_bilevel(combinedmodel, valloader, "Val", Y, weights, criterion, args_model_init["joint"])
        micro_f, macro_f, per_label_macro_f = eval_bilevel(combinedmodel, testloader, "Test", Y, weights, criterion, args_model_init["joint"])
        test_f.append((micro_f, macro_f, t+1))
        if macro_val > best_macro:
            best_macro = macro_val
            bests["macro"] = (micro_val, macro_val, t + 1)
        if micro_val > best_micro:
            best_micro = micro_val
            bests["micro"] = (micro_val, macro_val, t + 1)
        print(f"Total loss: {total_loss}")
        with open(exp_name+"/weights.txt",'a') as f:
            string = ", ".join(list(map(lambda x: str(x.item()),weights)))
            str_to_write = f"Epoch {t+1}/{epochs} \n" + string + "\n"
            f.write(str_to_write)
        with open(exp_name + "/f1scores.txt",'a') as f:
            string = "\n".join(list(map(lambda x: str(x.item()),per_label_macro_f)))
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
    parser.add_argument('--joint', default=False, action='store_true')
    parser.add_argument('--geodesic_lambda', default=0.1, type=float)
    parser.add_argument('--cascaded_step2', default=False, action='store_true')
    parser.add_argument('--pretrained_label_model', default=None)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--drop_prob', default=0.0, type=float)
    parser.add_argument('--dataset_size', default=20000, type=int)
    args = parser.parse_args()


    os.makedirs(args.exp_name, exist_ok=True)

    logging.basicConfig(filename=args.exp_name+'/res.txt', level=logging.DEBUG)
    logging.info(args)

    trainvalset = TextLabelDataset(args.drop_prob, args.dataset_size)

    # Split into train and val sets
    trainset, valset = torch.utils.data.dataset.random_split(trainvalset, 
                [int(0.9*len(trainvalset)), len(trainvalset)- int(0.9*len(trainvalset))])

    trainloader = DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    valloader = DataLoader(
        valset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    testset = TextLabelDataset(0, int(2/3*args.dataset_size))
    
    testloader = DataLoader(
        testset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
    )


    emb_dim = 21  # Document and label embed length

    doc_lr = 0.001
    args_model_init = {
            "n_labels":trainvalset.n_labels,
            "lr" : doc_lr,
            "emb_dim" : emb_dim,
            "drop_p_doc" : 0.1,
            "drop_p_label" : 0.6,
            "flat" : args.flat,
            "joint" : args.joint
        }
    # Models
    combinedmodel = CombinedModel(args_model_init)
    # if args.cascaded_step2:
    #     label_model_pretrained = torch.load(args.pretrained_label_model)['label_model']
    #     label_model.load_state_dict(label_model_pretrained)

    # if args.flat or args.cascaded_step2:
    #     for param in label_model.parameters():
    #         param.require_grad = False

    combinedmodel = combinedmodel.cuda()

    # Loss and optimizer
    criterion = BiLevelLoss(use_geodesic=args.joint, only_label=args.cascaded_step1)

    optimizer = torch.optim.Adam(combinedmodel.parameters(),doc_lr)


    logging.info('Starting Training')
    # Train and evaluate
    Y = torch.arange(trainvalset.n_labels).cuda()
    train_bilevel(
            args.epochs,
            trainloader,
            valloader,
            testloader,
            combinedmodel,
            args_model_init,
            Y,
            optimizer,
            criterion,
            args.exp_name,
            save_folder='checkpoints',
            wt_lr= 0.1
        )
