import pickle
import torch
from tqdm import tqdm
from models import CombinedModel, LabelEmbedModel, TextCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Labels
import sys

testset = pickle.load(open("rcv1/test.pkl", "rb"))
labels_ = Labels('rcv1/rcv1_labels.txt')
itos = {v:k for k,v in labels_.stoi.items()}
print(itos)
vocab = testset.text_dataset.vocab
rev_vocab = {v:k for k,v in vocab.items()}

checkpoints = sys.argv[1]

for t in checkpoints:
    print(t)

    checkpoint = torch.load(t)

    test_data_loader = DataLoader(
        testset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True
    )

    args_model_init = {
        "n_labels":testset.n_labels,
        "lr" : 0.001,
        "vocab" : testset.text_dataset.vocab,
        "glove_file" : None,
        "emb_dim" : 300,
        "drop_p_doc" : 0.1,
        "word_embed_dim" : 300,
        "drop_p_label" : 0.6,
        "flat" : True
    }

    # label_model = LabelEmbedModel(testset.n_labels, emb_dim=300, dropout_p=0.1)
    # doc_model = TextCNN(
    #             testset.text_dataset.vocab,
    #             emb_dim=300,
    #             dropout_p=0.1,
    #             word_embed_dim=300,
    #         )

    # label_model = nn.DataParallel(label_model)
    # doc_model = nn.DataParallel(doc_model)

    # label_model.load_state_dict(checkpoint['label_model'])
    # doc_model.load_state_dict(checkpoint['doc_model'])

    # label_model = label_model.cuda()
    # doc_model = doc_model.cuda()

    # label_model = label_model.eval()
    # doc_model = doc_model.eval()
    combinedmodel = CombinedModel(args_model_init)
    combinedmodel = nn.DataParallel(combinedmodel)
    combinedmodel.load_state_dict(checkpoint['combinedmodel'])
    combinedmodel = combinedmodel.cuda()
    combinedmodel.eval()

    Y = torch.arange(testset.n_labels).cuda()

    tp, fp, fn = 0, 0, 0
    num_batch = test_data_loader.__len__()
    total_loss = 0.
    pbar = tqdm(total=num_batch)
    with torch.no_grad():
        for data in test_data_loader:
            pbar.update(1)

            docs, labels, edges = data
            docs, labels, edges = docs.cuda(), labels.cuda(), edges.cuda()
            doc_emb, label_emb, _ = combinedmodel(docs, Y, edges)
            dot = doc_emb @ label_emb.T
            t = torch.sigmoid(dot)

            y_pred = 1.0 * (t > 0.5)
            y_true = labels

            tp += (y_true * y_pred).sum(dim=0)
            fp += ((1 - y_true) * y_pred).sum(dim=0)
            fn += (y_true * (1 - y_pred)).sum(dim=0)

        eps = 1e-7
        p = tp.sum()/(tp.sum() + fp.sum() + eps)
        r = tp.sum()/(tp.sum() + fn.sum() + eps)
        micro_f = 2*p*r/(p+r+eps)
        macro_p = tp/(tp+fp+eps)
        macro_r = tp/(tp+fn+eps)
        macro_f = (2*macro_p*macro_r/(macro_p + macro_r + eps)).mean()
    pbar.close()

    print(2*macro_p*macro_r/(macro_p + macro_r + eps))
