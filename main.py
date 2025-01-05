# -*- coding: utf-8 -*-
import json
import logging
import math
import os
import time
from multiprocessing import Queue
import torch
from torch import nn
import multiprocessing

class DecoderLayer(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward )
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self,x,attn_mask):
        r=x
        x=self.self_attn(x,x,x,attn_mask=attn_mask)[0]
        x=self.dropout(x)
        x=x+r
        x=self.norm1(x)
        r=x
        x =self.linear1(x)
        x=self.activation(x)
        x=self.dropout1(x)
        x=self.linear2(x)
        x=self.dropout2(x)
        x=x+r
        x=self.norm2(x)
        return x
class WYY (nn.Module):
    def __init__(self,d_model,vocab_size,nlayer,nhead,max_len,batch_size,dim_feedforward,pad_id,device):
        super(WYY, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.position=torch.concat([torch.arange(0,max_len-1).unsqueeze(0) for _ in range(batch_size)],dim=0).to(device)
        self.Decoders= self.decoder_block(nlayer, d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.atten_mask = torch.triu(torch.ones(size=(max_len-1, max_len-1), device=device) * float('-inf'),
                                     diagonal=1).to(device)


    def decoder_block(self,n,d_model,nhead,dim_feedforward):
        return nn.ModuleList([DecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward) for _ in range(n)])
    def forward(self, x):
        p=self.position_embedding(self.position)
        x=self.embedding(x)
        x=x+p
        for Decoder in self.Decoders:
            x=Decoder(x,self.atten_mask)
        x=self.fc_out(x)
        return x

class Dict(object):
    def __init__(self):
        self.vocab_path='vocab.json'
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
    def vocab_to_id(self,data,max_length):
        datas=[]
        temp=[self.vocab['<Q>']]
        for words in data:
            for word in words:
                    temp.append(self.vocab.get(word,self.vocab['<UNK>']))
            temp.append(self.vocab['<END>'])
            temp.append(self.vocab['<A>'])
        temp.pop(-1)
        d= self.pad_or_crop(temp,max_length)
        for i in d:
            datas.append(i)
        return datas
    def pad_or_crop(self,datas,max_length):
        d=[]
        b=0
        e=max_length
        while True:
            if e>len(datas):
                temp=datas[b:]
                temp.extend([self.vocab["<PAD>"]] * (max_length - (len(datas) - b)))
                d.append(temp)
                break
            elif e<len(datas):
                d.append(datas[b:e])
            else:
                d.append(datas[b:e])
            b+=max_length
            e+=max_length
        return d
    def make_vocab(self,datas_path):
        paths=[os.path.join(datas_path,path) for path in os.listdir(datas_path)]
        vocab_num={k:[v,0] for k,v in self.vocab.items()}
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    for word in line:
                        if word not in vocab_num:
                            vocab_num[word]=[len(vocab_num),1]
                        else:
                            vocab_num[word][1]+=1
        vocab_num={k:v for k,v in vocab_num.items() if v[1]>1000}
        i=0
        for k,v in vocab_num.items():
            vocab_num[k]=[i,v[1]]
            i+=1
        self.vocab={k:v[0] for k,v in vocab_num.items()}
        self.save(self.vocab_path,self.vocab)
        vocab_num={v[0]:v[1] for k,v in vocab_num.items()}
        self.save('vocab_num.json',vocab_num)
    def save(self,path,dic):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
class DataDeal(object):
    def __init__(self,max_len):
        self.dict = Dict()
        self.max_len=max_len
    def data_deal(self,lines,mode=None):#mode="json"):
        datas=[]
        data=[]
        if mode=="json":
            for line in lines:
                data.append( json.loads(line)['content'])

            data_id= self.dict.vocab_to_id(data,self.max_len)
            for i in range(len(data_id)):
                datas.append(data_id[i])
        else:
            data="".join(lines)
            data = self.dict.vocab_to_id(data, self.max_len)
            for i in range(len(data)):
                datas.append(data[i])
        return datas
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len,data_deal):
        self.datas = data_deal.data_deal(data)
        self.datas=torch.tensor(self.datas,dtype=torch.long)
        self.lables = self.datas[:,1:]
        self.datas = self.datas[:,:-1]
        self.max_len = max_len
    def __getitem__(self, index):
        return self.datas[index],self.lables[index]
    def __len__(self):
        return len(self.datas)
class DatasetQueue(object):
    def __init__(self, path, max_len,max_queue_size=10):
        self.queue = Queue(maxsize=max_queue_size)
        self.path = path
        self.max_len = max_len
        self.max_queue_size=max_queue_size
    def put(self,queue,stop_event):
        paths = os.listdir(self.path)
        paths = [os.path.join(self.path, path) for path in paths]
        datas = []
        data_deal = DataDeal(self.max_len)
        for path in paths:
            end = False
            with open(path, 'r', encoding='utf-8') as f:
                while True:
                    for _ in range(1000):
                        data = f.readline()
                        if not data:
                            end = True
                            break
                        elif len(datas)==999:
                            datas.append(data)
                            break
                        datas.append(data)
                    if len(datas) >= 1000:
                        dataset = Dataset(datas, max_len=self.max_len, data_deal=data_deal)
                        queue.put(dataset)
                        datas = []
                    if end:
                        break
        queue.put(None)
        while not stop_event.is_set():
            time.sleep(1)
class DataLoaders(object):
    def __init__(self, batch_size, dataset_queue,shuffle=True,drop_last=True):
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.drop_last=drop_last
        self.dataset_queue=dataset_queue

    def get_dataloader(self):
        while True:
            dataset=self.dataset_queue.get()
            if dataset is None:
                break
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                              drop_last=self.drop_last,num_workers=0,pin_memory=True)
            yield dataloader
def save_checkpoint(model,optimizer,scheduler,epoch,data_num):
    state = {
        'epoch': epoch,
        'data_num': data_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(state, 'checkpoint.pth')
class Logger(object):
    def __init__(self):
        self.logger=logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler('app.log',encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
    def debug(self,message):
        self.logger.debug(message)
    def info(self,message):
        self.logger.info(message)
    def warning(self,message):
        self.logger.warning(message)
    def error(self,message):
        self.logger.error(message)
    def critical(self,message):
        self.logger.critical(message)
def make_vocab_weight(path):
    with open(path, 'r', encoding='utf-8') as f:
        vocab_num=json.load(f)
    vocab_weight=[1/math.log(v+1) for v in vocab_num.values()]
    vocab_weight=torch.tensor(vocab_weight)
    torch.save(vocab_weight,"vocab_weight.pt")
def train(path,nlayer,nhead,total_steps,vocal_size,d_model,max_len,batch_size,target_batch_size,epoch,dim_feedforward,pad_id,max_queue_size,device, weight_decay=0, clip_norm=1.0,label_smoothing=0.3):
    logger=Logger()
    model=WYY(vocab_size=vocal_size,nlayer=nlayer,nhead=nhead,d_model=d_model,max_len=max_len,batch_size=batch_size,dim_feedforward=dim_feedforward,pad_id=pad_id,device=device)
    model.train()
    model.to(device)
    vocab_weight =torch.load("vocab_weight.pt",weights_only=True)
    vocab_weight = vocab_weight.to(device)
    loss=nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_id,label_smoothing=label_smoothing,weight=vocab_weight,)
    loss.to(device)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=0.0000001,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.000001,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=1e4
    )
    initial_clip_norm = clip_norm
    current_clip_norm = initial_clip_norm
    gradient_norms = []
    data_num=0
    start_epoch=0
    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth",weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        data_num = checkpoint['data_num']

    loss_list=[]
    for epoch_ in range(start_epoch,epoch):
            logger.info("开始第{}轮训练".format(epoch_))
            dataset_queue=DatasetQueue(path=path,max_len=max_len,max_queue_size=max_queue_size)
            stop_event = multiprocessing.Event()
            multiprocessing.Process(target=dataset_queue.put,args=(dataset_queue.queue,stop_event)).start()
            dataloaders=DataLoaders(batch_size=batch_size,dataset_queue=dataset_queue.queue)
            logger.info("成功创建数据集生成器")
            get_dataloader=dataloaders.get_dataloader()
            optimizer.zero_grad()
            for _ in range(data_num):
                next(get_dataloader)
            for dataloader in get_dataloader:
                grad_num=0
                for data in dataloader:
                    data,label=data[0].to(device),data[1].to(device)
                    r = model(data)
                    r = r.view(-1, r.shape[-1])
                    y=label.view(-1)
                    l = loss(r, y)
                    loss_list.append(l.item())
                    l.backward()
                    grad_num+=1
                    if  grad_num % (target_batch_size // batch_size) == 0:
                        logger.info("完成损失计算损失为{}".format(sum(loss_list)/len(loss_list)))
                        loss_list=[]
                        global_norm=nn.utils.clip_grad_norm_(model.parameters(), current_clip_norm)
                        gradient_norms.append(global_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        if len(gradient_norms) >= 100:
                            avg_norm = sum(gradient_norms) / 100
                            if avg_norm > current_clip_norm * 1.5:
                                current_clip_norm *= 1.1
                            elif avg_norm < current_clip_norm * 0.5:
                                current_clip_norm /= 1.1
                            gradient_norms = gradient_norms[10:]
                data_num+=1
                save_checkpoint(model,optimizer,scheduler,epoch_,data_num)
                logger.info("完成checkpoint保存")
            data_num=0
            stop_event.set()
    logger.info("完成训练")
def predict(nlayer,nhead,vocal_size,d_model,max_len,batch_size,dim_feedforward,pad_id,device):
    model=WYY(vocab_size=vocal_size,nlayer=nlayer,nhead=nhead,d_model=d_model,max_len=max_len,batch_size=batch_size,dim_feedforward=dim_feedforward,pad_id=pad_id,device=device)
    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth",weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    text="A:明天会下雨吗？\nB:"
    with open("vocab.json", 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_vocab = {v: k for k, v in vocab.items()}
    data=[vocab["<Q>"]]
    for word in text:
        data.append(vocab.get(word))
    l=len(data)
    if l<max_len:
        data.extend([vocab["<PAD>"]]*(max_len-len(data)))
    elif l>=max_len:
        return False
    data=[data]
    data=torch.tensor(data)[:,:-1]
    data=torch.cat([data for _ in range(batch_size)])
    data=data.to(device)
    p=torch.tensor([0]*batch_size).reshape(5,1).to(device)
    softmax=torch.nn.Softmax(dim=1)
    end=False
    for _ in range(max_len-l-1):
        if end:
            break
        output=model(data)[:,l-1,:]
        output=softmax(output)
        output=output.detach()
        topk_p,topk_id=torch.topk(output,batch_size)
        topk_p=torch.log(topk_p)+p
        topk_p=topk_p.reshape(-1,)
        topk_id=topk_id.reshape(-1,)
        topk_p_,topk_id_=torch.topk(topk_p,batch_size)
        p=topk_p_.reshape(5,1)
        w=[]
        for i in range(batch_size):
            w.append(data[topk_id_[i]//batch_size].unsqueeze(0))
            char_id=topk_id[topk_id_[i]]
            w[i][0][l]=char_id
            if char_id==vocab["<END>"]:
                end=True
                break
        data=torch.cat(w,dim=0)
        l+=1
    word=data[torch.topk(p.reshape(-1,),1)[1]].squeeze(0)
    for i in range(max_len-1):
        print(id_to_vocab[word[i].item()],end="")
if __name__ == '__main__':
    predict(vocal_size=4215,d_model=512,max_len=512,batch_size=5,dim_feedforward=1024,pad_id=0,nlayer=12,nhead=8,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    # train(path="D:\code\datas\\1623_0000001\zh",total_steps=125492,vocal_size=4215,d_model=512,max_len=512,batch_size=12,target_batch_size=60,epoch=2,dim_feedforward=1024,pad_id=0,max_queue_size=200,nlayer=12,nhead=8,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    # train(path="D:\code\datas\晴数智慧MagicData大模型训练数据集\TXT",total_steps=366,vocal_size=4215,d_model=512,max_len=512,batch_size=12,target_batch_size=60,epoch=2,dim_feedforward=1024,pad_id=0,max_queue_size=100,nlayer=12,nhead=8,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
