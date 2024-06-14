import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from my_transformer import TransformerEncoder, TransformerEncoderLayer
import math, copy
import random

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class TransformerModel(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.maxpooling = nn.MaxPool2d(128, 1)

    def forward(self, q, k, v):
        #query, key, value = self.encoder(query), self.encoder(key), self.encoder(value) 
        output = self.transformer_encoder(q, k, v)
        max_x, _ = output.max(dim=1, keepdim=True)
        return max_x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PRM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PRM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.softmax(x, dim=1)
        return out

class SCL(th.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):
        # In-domain Contrastive Learning when label_2 is None. 
        inrep_1.to(device)
        inrep_2.to(device)
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 is None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = th.diag(cosine_similarity)
            cos_diag = th.diag_embed(diag)  # bs,bs

            label = th.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = th.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs



                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()

                cosine_similarity = (cosine_similarity-cos_diag) / self.temperature  # the diag is 0
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte(), -float('inf'))  # mask the diag

                cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat

                cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs
        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)
            label_1_mat = th.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)
            label_2_mat = th.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat #find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)

        cos_loss = -th.mean(cos_loss, dim=0)
        return cos_loss

class Net(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, alpha=[0.5, 0.5], beta=[0.5, 0.5], gama=[0.8, 0.1, 0.1]):
        super(Net, self).__init__()
        self.transformer = TransformerModel(d_model, nhead, num_layers)
        self.prm = PRM(d_model, 256, 2)
        self.cl_module = SCL()
        self.alpha = alpha
        self.beta = beta
        self.gama = gama  # to reweight the losses. 
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, in_data, out_data):
        # out_data's labels are not available. 
        # in/out_data = [Batch_size, Seq_len, E]
        in_x = self.transformer(q=in_data.x, k=in_data.x, v=in_data.x)
        out_x = self.transformer(q=out_data.x, k=out_data.x, v=out_data.x)
        # print('Transformer output shape: ', in_x.shape, out_x.shape)
        # in_x = [Batch_size, 1, E]
        in_x.squeeze_(dim=1) # [Batch_size, E]
        out_x.squeeze_(dim=1) # [Batch_size, E]
        in_pred = self.prm(in_x)
        
        copied_in_x = in_x
        in_y = in_data.y

        # Prototype Contrastive Learning
        prototype_cl_loss = 0
        in_centriods, centriod_unique_labels = self.compute_label_centroids(copied_in_x, in_y)
        if centriod_unique_labels.shape[0] == 2:
            out_pseudo_labels = self.assign_labels_to_unlabelled(in_centriods, out_x)
            # print(out_pseudo_labels, centriod_unique_labels.shape[0]==2)
            prototype_cl_loss = self.cl_module(out_x, in_centriods, out_pseudo_labels, centriod_unique_labels)
        # In-domain Contrastive Learning
        in_cl_loss = self.cl_module(in_x, in_x, in_y)
        out_cl_loss = self.cl_module(out_x, out_x, out_pseudo_labels)
        
        # Cross-domain Contrastive Learning
        cross_cl_loss1 = self.cl_module(in_x, out_x, in_y, out_pseudo_labels)
        cross_cl_loss2 = self.cl_module(out_x, in_x, out_pseudo_labels, in_y)
        


        # Cross-Attention 
        cross_attn_out = self.transformer(q=in_data.x, k=out_data.x, v=out_data.x)
        cross_attn_out.squeeze_(dim=1)
        cross_pred_out = self.prm(cross_attn_out)
        kl_loss =  F.kl_div(F.log_softmax(in_pred, dim=1), F.softmax(cross_pred_out, dim=1), reduction='batchmean')

        intro_cl_loss = self.alpha[0] * in_cl_loss + self.alpha[1] * out_cl_loss
        cross_cl_loss = cross_cl_loss1 + cross_cl_loss2 + prototype_cl_loss
        all_cl_loss = self.beta[0]*intro_cl_loss + self.beta[1]*cross_cl_loss

        ce_loss = self.ce_loss(in_pred, in_y)/in_x.shape[0]

        #print('Visualisation of all losses:')
        #print(ce_loss, all_cl_loss, kl_loss)
        #print('cl_loss: ',in_cl_loss, out_cl_loss, cross_cl_loss1, cross_cl_loss2, prototype_cl_loss)
        #print('kl_loss: ', kl_loss)
        all_loss = self.gama[0]*ce_loss + self.gama[1]*all_cl_loss + self.gama[2]*kl_loss
        return in_pred, all_loss


    def predict(self, data):
        x = self.transformer(q=data.x, k=data.x, v=data.x)
        x.squeeze_(dim=1)
        return self.prm(x)

    def compute_label_centroids(self, embeddings, labels):
        embeddings = torch.tensor(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings
        labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        unique_labels = labels.unique()  # This is sorted from 0 to 1. 
        centroids = []
        for label in unique_labels:
            label_embeddings = embeddings[labels == label]
            centroid = label_embeddings.mean(dim=0)
            centroids.append(centroid)
        centroids = torch.stack(centroids)
        return centroids, unique_labels
        
    def assign_labels_to_unlabelled(self, centroids, unlabelled_embeddings):
        unlabelled_embeddings = torch.tensor(unlabelled_embeddings) if not isinstance(unlabelled_embeddings, torch.Tensor) else unlabelled_embeddings
        dists = torch.cdist(unlabelled_embeddings, centroids)
        labels = torch.argmin(dists, dim=1)
        return labels
    
                