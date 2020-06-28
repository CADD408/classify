# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:10:34 2020

@author: 86136
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_csv = pd.read_csv(r"C:\Users\86136\Desktop\CYSLTR1_DATA\CYSLT1_standerd_database.csv")
data_maccs = pd.read_csv(r"C:\Users\86136\Desktop\CYSLTR1_DATA\cysltr1_maccs_1.csv")
smiles_list = []
labels = []
for i in range(len(data_csv)):
    smiles = data_csv['SMILES'][i]
    smiles = smiles.replace('Cl','Q')#用Q来代替Cl
    smiles = smiles.replace('Br','W')#用W来代替Br
    smiles = smiles.replace('Na+','V')#用V来代替Na
    smiles = smiles.replace('NaH','V')
    smiles_list.append([ c for c in smiles] )
    labels.append(data_csv['new_value'][i])

##########################################################################################

#创建一个字符集合
chars = set()
 
for smi in smiles_list:
    chars = chars.union(set(char for char in smi))#两个集合取并集
chars = sorted(list(chars))
#字符对应数字
smiles2index = dict((char, i+1) for i,char in enumerate(chars))

#数字对应字符
index2smiles = dict((i+1,char) for i,char in enumerate(chars))
#index2smiles[(len(chars)+1)]="<EOS>"

chars_size=len(smiles2index)
max_length = max(len(s) for s in smiles)


def encode(smiles_list,smiles_dict):
    length = []

    out_smiles = [[smiles_dict.get(c,0) for c in smiles] for smiles in smiles_list]#如果没有字典里没有该元素，返回None
    length = [len(smiles) for smiles in smiles_list]

    return out_smiles,length

X,len_X  = encode(smiles_list, smiles2index)
Y = labels
#######################################################################################
print(X[0])



###########################################################################################
def get_minibatches(n, minibatch_size, shuffle = False):#如果shuffle = true 是打乱batch之间的顺序,n是总数
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for index in idx_list:
        minibatches.append(np.arange(index, min(index + minibatch_size, n)))
    return minibatches

#将smiles长度
def prepare_data(smiles_list):
    lengths = [len(smile) for smile in smiles_list]
    n_sample = len(smiles_list)
    max_len = np.max(lengths)
    
    x = np.zeros((n_sample, max_len)).astype('float32')
    x_len = np.array(lengths).astype('float32')
    for index, smile in enumerate(smiles_list):
        x[index, :lengths[index]] = smile
    return x, x_len

def gen_examples(smiles, batch_size):#以所有数据作为一个batch进行长短排序
    minibatches = get_minibatches(len(smiles), batch_size)
    all_ex = []
    all_ex_len = []
    for minibatch in minibatches:
        mb_smiles = [smiles[i] for i in minibatch]    
        #mb_labels = [label[i] for i in minibatch]
        mb_x, mb_x_len = prepare_data(mb_smiles)
        #mb_y, mb_y_len = prepare_data(mb_labels)
        all_ex_len.append(mb_x_len)
        all_ex.append((mb_x))
    return all_ex,all_ex_len


X_data,seq_len_list = gen_examples(X, len(X))
seq_len_list = seq_len_list[0]
MAX_LENGTH = int(max(seq_len_list))


#########################################################
#对y进行one_hot编码
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder(categories='auto')  
enc.fit([[0],[1],[0],[1]])  #这里一共有2个数据，2种特征
Y_data = np.array(Y)
Y_data = Y_data.reshape(-1,1)
Y_data = enc.transform(Y_data).toarray()

####################################################################
#用于以后求出每一个化合物的attention weight
for x_array in X_data:
    x_all = torch.from_numpy(x_array)
    label = np.array(Y_data)
    label = torch.from_numpy(label)
    trainset = torch.utils.data.TensorDataset(x_all,label)
    ALLloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=False, num_workers=0)

########################################################
#分训练集测试集，划分成batch，每个batch一个化合物
for x_array in X_data:
    X_train, X_test, y_train_0, y_test_0 = train_test_split( x_array ,Y_data, test_size = 0.2, random_state = 223)
    X_train_1 = torch.from_numpy(X_train)
    X_test_1 = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train_0)
    y_test = torch.from_numpy(y_test_0)

    trainset = torch.utils.data.TensorDataset(X_train_1,y_train)
    testset = torch.utils.data.TensorDataset(X_test_1,y_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= 1, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size= 1, shuffle=True, num_workers=0)



#####################################################################################################
class GRU_attention(nn.Module):
    def __init__(self,hidden_size ,vocab_size , embed_size):
        super(GRU_attention, self).__init__()
        self.hidden_dim = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 双向GRU，//操作为了与后面的Attention操作维度匹配，hidden_dim要取偶数！
        self.bigru = nn.GRU(embed_size, self.hidden_dim // 2, num_layers=2, bidirectional=True)
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        # 二分类
        self.fc = nn.Linear(self.hidden_dim, 2)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1,-1,1) # [seq_len, bs, emb_dim]
        embeds = embeds.permute(1,0,2)
        gru_out, _ = self.bigru(embeds) # [seq_len, bs, hid_dim]
        x = gru_out.permute(1, 0, 2) # 更换维度
        # # # Attention过程，与上图中三个公式对应
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        # # # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1) # sum求和，1是按行求和
        y = self.fc(feat)

        out = F.softmax(y,dim=1)
        return out,att_score





embed_size = 1
vocab_size = (chars_size + 1)
hidden_size = 128
output_size = 2
AttRNN = GRU_attention(hidden_size,vocab_size,embed_size)
learning_rate = 0.0005
epoch = 25
optimizer = torch.optim.Adam(AttRNN.parameters(), lr = learning_rate)
criterion = nn.BCELoss()
train_loss_list = []
test_loss_list = []
test_acc = []
test_pred = []
test_true = []
for m in range(epoch):
    training_loss = 0.0
    for index, data in enumerate(trainloader):

        AttRNN.train()
        AttRNN.zero_grad()
        inputs, labels = data

        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long() 
        labels = Variable(labels).float()

        optimizer.zero_grad()      
        output,att_weight = AttRNN(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    m_train_loss_mean = training_loss/len(X_train)
    train_loss_list.append(m_train_loss_mean)
    #训练log的保存地址
    PATH = r"C:\Users\86136\Desktop\logs\att_rnn"+str(m)+r".pth"
    torch.save(AttRNN, PATH)
    with torch.no_grad():
        AttRNN.eval()
        total_test = 0
        correct_test = 0
        total_train = 0
        correct_train = 0
        for index, data in enumerate(testloader):
             inputs, labels = data
             labels = Variable(labels).float()
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()
             output,att_weight = AttRNN(inputs)

             _,predicted = torch.max(output.detach(), 1)#1代表行，0代表列，返回每一行的最大值,.detach()表示返回output 的值且且 requires_grad=False
             _,labels = torch.max(labels.detach(), 1)
             total_test += labels.size(0)
             correct_test += (predicted == labels).sum().item()
        for index, data in enumerate(trainloader):
             inputs, labels = data
             labels = Variable(labels).float()
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()
             output,att_weight = AttRNN(inputs)

             _,predicted = torch.max(output.detach(), 1)
             _,labels = torch.max(labels.detach(), 1)
             total_train += labels.size(0)
             correct_train += (predicted == labels).sum().item()        
        acc_test = 100*correct_test//total_test
        acc_train = 100*correct_train//total_train

    print(m, 'epochs test acc:', acc_test,"train acc:",acc_train)
print("finish",m,"epoch")



##########
#二次训练

model = torch.load(r"C:\Users\86136\Desktop\logs\att_rnn24.pth")
optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.0002)

epoch_1 = 20
for m in range(epoch_1):
    training_loss = 0.0
    for index, data in enumerate(trainloader):

        inputs, labels = data

        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long() 
        labels = Variable(labels).float()

        optimizer_1.zero_grad()      
        output,att_weight = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer_1.step()
        training_loss += loss.item()

    m_train_loss_mean = training_loss/len(X_train)
    train_loss_list.append(m_train_loss_mean)
    print(m,"epoch train loss:",m_train_loss_mean)
    PATH = r"C:\Users\86136\Desktop\logs\att_rnn"+"\\25_"+str(m)+r".pth"
    torch.save(model, PATH)
    with torch.no_grad():
        model.eval()
        total_test = 0
        correct_test = 0
        total_train = 0
        correct_train = 0
        for index, data in enumerate(testloader):
             inputs, labels = data
             labels = Variable(labels).float()
#             print('labels:',labels)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()
             output,att_weight = model(inputs)

             _,predicted = torch.max(output.detach(), 1)#1代表行，0代表列，返回每一行的最大值,.detach()表示返回output 的值且且 requires_grad=False
             _,labels = torch.max(labels.detach(), 1)
             total_test += labels.size(0)
             correct_test += (predicted == labels).sum().item()
        for index, data in enumerate(trainloader):
             inputs, labels = data
             labels = Variable(labels).float()
#             print('labels:',labels)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()
             output,att_weight = model(inputs)

             _,predicted = torch.max(output.detach(), 1)#1代表行，0代表列，返回每一行的最大值,.detach()表示返回output 的值且且 requires_grad=False
             _,labels = torch.max(labels.detach(), 1)
             total_train += labels.size(0)
             correct_train += (predicted == labels).sum().item()        
        acc_test = 100*correct_test//total_test
        acc_train = 100*correct_train//total_train

    print(m, 'epochs test acc:', acc_test,"train acc:",acc_train)    
    
    
    
##########################################################################################
#验证模型
######################################################################
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
#验证模型
model_1 = torch.load(r"C:\Users\86136\Desktop\logs\att_rnn\25_18.pth")
print(model_1)
#model_1.eval()
with torch.no_grad():
    test_pred = []
    test_true = []
    roc_test_score = []
#########测试集指标
    for inputs, labels in testloader:
         inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()       
         labels = Variable(labels).float()
         output,att_weight = model_1(inputs)
         outputs = output.detach().numpy().tolist()
         for x in outputs:
             y_score = x[-1]
             roc_test_score.append(y_score)
         _,predicted = torch.max(output.detach(), 1)#1代表行，0代表列，返回每一行的最大值,.detach()表示返回output 的值且且 requires_grad=False
         _,labels = torch.max(labels.detach(), 1)
         pred_list = predicted.detach().numpy().tolist()
         for x in pred_list:
             test_pred.append(x)
         true_list = labels.detach().numpy().tolist()
         for x in true_list:
             test_true.append(x)
    print(sklearn.metrics.classification_report(test_true, test_pred, digits=4, target_names = ["0","1"]))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_true, roc_test_score, pos_label=1)
    AUC = sklearn.metrics.auc(fpr, tpr)
    plt.figure(figsize=(12,9))
    plt.plot(fpr,tpr,color='darkorange',label='Test ROC curve (area = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    
    MCC =  matthews_corrcoef(test_true, test_pred)
    print("test MCC:",MCC)
################训练集
    train_pred = []
    train_true = []
    roc_train_score = []
    for inputs, labels in trainloader:
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()       
        labels = Variable(labels).float()
        output,att_weight = model_1(inputs)
        outputs = output.detach().numpy().tolist()
        for x in outputs:
            y_score = x[-1]
            roc_train_score.append(y_score)
        _,predicted = torch.max(output.detach(), 1)#1代表行，0代表列，返回每一行的最大值,.detach()表示返回output 的值且且 requires_grad=False
        _,labels = torch.max(labels.detach(), 1)
        pred_list = predicted.detach().numpy().tolist()
        for x in pred_list:
            train_pred.append(x)
        true_list = labels.detach().numpy().tolist()
        for x in true_list:
            train_true.append(x)
    print(sklearn.metrics.classification_report(train_true, train_pred,digits=4, target_names = ["0","1"]))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_true, roc_train_score, pos_label=1)
    AUC = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr,tpr,color='red',label='Train ROC curve (area = %0.4f)' % AUC)
    plt.legend(loc="lower right")

    plt.savefig(r"C:\Users\86136\Desktop\result\ROC_smiles_RNN.png")
    plt.show()
    MCC_train =  matthews_corrcoef(train_true, train_pred)
    print("train MCC:",MCC_train)
  
####
#生成每一个化合物的attention weight    
latent_list = []
with torch.no_grad():
    for index, data in enumerate(ALLloader):      
        inputs, labels = data
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).long()
        output,att_weight = model_1(inputs)
        att_weight = att_weight.detach().numpy().tolist()
        for x in att_weight:
            weight = x[:int(seq_len_list[index])]
            latent_list.append(weight)
for i in range(len(latent_list)):
    df = pd.DataFrame({'smiles':smiles_list[i],'weight':latent_list[i]})
    path = r"C:\Users\86136\Desktop\result\RNN_weight" +'\\'+str(i)+'_weight.csv' 
    df.to_csv(path)   

dele = ["#","(",")","-",".","/","1","2","3","4","5","6","7","=","@","[","\\","]","H"]
num_del = []
for i in range(503):
    path = r"C:\Users\86136\Desktop\result\RNN_weight\old weight" +'\\'+str(i)+'_weight.csv'
    data = pd.read_csv(path)
    num_del.clear()
    for m in range(len(data)):
        if data["smiles"][m] in dele:
            num_del.append(m)
    data_new = data.drop(num_del)
    data_new.columns = ['ID','smiles','IC50']
    data_new = data_new.drop(['ID'],axis = 1)
    data_new.index = range(len(data_new))
    data_new.to_csv(r"C:\Users\86136\Desktop\result\RNN_weight" +'\\'+str(i)+'_new_weight.csv')



###################################################################################################
####画图，画出每个化合物的前10个重要元素    
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import pandas as pd
data_csv= pd.read_csv(r"C:\Users\86136\Desktop\CYSLTR1_DATA\CYSLT1_standerd_database.csv")
Chem.PandasTools.AddMoleculeColumnToFrame(data_csv, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
opts = Draw.DrawingOptions()
opts.elemDict = {1: (0, 0, 0),
 7: (0, 0, 0),
 8: (0, 0, 0),
 9: (0, 0, 0),
 15: (0, 0, 0),
 16: (0, 0, 0),
 17: (0, 0, 0),
 35: (0, 0, 0),
 53: (0, 0, 0),
 0: (0, 0, 0)}


for i in range(len(data_csv)):
    if data_csv['new_value'][i] == 1:
        #每一个化合物的attention weight
        path = r"C:\Users\86136\Desktop\result\RNN_weight" + '\\'+str(i)+'_new_weight.csv'  
        data = pd.read_csv(path)
        weight = data['IC50'].tolist()
        index_ = sorted(range(len(weight)), key=lambda x: weight[x], reverse = True)
        mol = data_csv['MOL'][i]
        img = Draw.MolToImage(mol,size=(800, 800), options = opts, highlightAtoms = index_[:len(index_)//3],highlightColor = [255,0,0])
        #img.show()
        path_smile = r"C:\Users\86136\Desktop\result\weight_mol\high" + '\\' +str(i) +'.png'
        img.save(path_smile)
    
        