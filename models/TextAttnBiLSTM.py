'''
@Author: Gordon Lee
@Date: 2019-08-16 13:34:15
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-17 01:49:03
@Description: 
'''
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from config import Config
from datasets import SSTreebankDataset
from utils import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, train, validate, testing

class ModelConfig():
    '''
    模型配置参数
    '''
    # 全局配置参数
    opt = Config()

    # 数据参数
    output_folder = opt.output_folder
    data_name = opt.data_name
    SST_path  = opt.SST_path
    emb_file = opt.emb_file
    emb_format = opt.emb_format
    output_folder = opt.output_folder
    min_word_freq = opt.min_word_freq
    max_len = opt.max_len

    # 训练参数
    epochs = 120  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 16 # batch_size
    workers = 4  # 多处理器加载数据
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5 # 权重衰减率
    decay_epoch = 15 # 多少个epoch后执行学习率衰减
    improvement_epoch = 30 # 多少个epoch后执行early stopping
    is_Linux = True # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint =  None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextAttnBiLSTM' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    embed_dropout = 0.3 # dropout
    model_dropout = 0.5 # dropout
    fc_dropout = 0.5 # dropout
    num_layers = 2 # LSTM层数
    embed_dim = 128 # 未使用预训练词向量的默认值
    use_embed = True # 是否使用预训练
    use_gru = True # 是否使用GRU
    grad_clip = 4. # 梯度裁剪阈值

class Attn(nn.Module):
    '''
    Attention Layer
    '''
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        '''
        :param x: (batch_size, max_len, hidden_size)
        :return alpha: (batch_size, max_len)
        '''
        x = torch.tanh(x) # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2) # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1) # (batch_size, 1, max_len)
        return alpha

class ModelAttnBiLSTM(nn.Module):
    '''
    BiLSTM: BiLSTM, BiGRU
    '''
    def __init__(self, vocab_size, embed_dim, hidden_size, pretrain_embed, use_gru, embed_dropout, fc_dropout, model_dropout, num_layers, class_num, use_embed):

        super(ModelAttnBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        if use_embed:
            self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(pretrain_embed, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dropout = nn.Dropout(embed_dropout)   
         
        if use_gru:
            self.bilstm = nn.GRU(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout), bidirectional=True, batch_first=True)
        else:
            self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout), bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, class_num)
        
        self.fc_dropout = nn.Dropout(fc_dropout) 

        self.attn = Attn(hidden_size)
        
    def forward(self, x):
        '''
        :param x: [batch_size, max_len]
        :return logits: logits
        '''
        x = self.embedding(x) # (batch_size, max_len, word_vec)
        x = self.embed_dropout(x)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions), 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x) # (batch_size, max_len, hidden_size*num_directions)
        y = y[:,:,:self.hidden_size] + y[:,:,self.hidden_size:] # (batch_size, max_len, hidden_size)
        alpha = self.attn(y) # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1) # (batch_size, hidden_size)
        h = torch.tanh(r) # (batch_size, hidden_size)
        logits = self.fc(h) # (batch_size, class_num)
        logits = self.fc_dropout(logits)
        return logits
        

def train_eval(opt):
    '''
    训练和验证
    '''
    # 初始化best accuracy
    best_acc = 0.

    # epoch
    start_epoch = 0
    epochs = opt.epochs
    epochs_since_improvement = 0  # 跟踪训练时的验证集上的BLEU变化，每过一个epoch没提升则加1

    # 读入词表
    word_map_file = opt.output_folder +  opt.data_name + '_' + 'wordmap.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 加载预训练词向量
    embed_file = opt.output_folder + opt.data_name + '_' + 'pretrain_embed.pth'
    embed_file = torch.load(embed_file)
    pretrain_embed, embed_dim = embed_file['pretrain'], embed_file['dim']

    # 初始化/加载模型
    if opt.checkpoint is None:
        if opt.use_embed == False: embed_dim = opt.embed_dim
        model = ModelAttnBiLSTM(vocab_size=len(word_map), 
                      embed_dim=embed_dim, 
                      hidden_size=embed_dim,
                      class_num=opt.class_num,
                      pretrain_embed=pretrain_embed,
                      num_layers=opt.num_layers,
                      model_dropout=opt.model_dropout, 
                      fc_dropout=opt.fc_dropout,
                      embed_dropout=opt.embed_dropout,
                      use_gru=opt.use_gru, 
                      use_embed=opt.use_embed)
    
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)
        
    else:
        # 载入checkpoint
        checkpoint = torch.load(opt.checkpoint, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_acc = checkpoint['acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    # 移动到GPU
    model = model.to(opt.device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(opt.device)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'train'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'dev'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        
        # 学习率衰减
        if epoch > opt.decay_epoch:
            adjust_learning_rate(optimizer, epoch)
        
        # early stopping 如果dev上的acc在6个连续epoch上没有提升
        if epochs_since_improvement == opt.improvement_epoch:
            break
        
        # 一个epoch的训练
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              vocab_size=len(word_map),
              print_freq=opt.print_freq,
              device=opt.device,
              grad_clip=opt.grad_clip)
        
        # 一个epoch的验证
        recent_acc = validate(val_loader=val_loader,
                              model=model,
                              criterion=criterion,
                              print_freq=opt.print_freq,
                              device=opt.device)
        
        # 检查是否有提升
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        # 保存模型
        save_checkpoint(opt.model_name, opt.data_name, epoch, epochs_since_improvement, model, optimizer, recent_acc, is_best)

def test(opt):

    # 载入best model
    best_model = torch.load(opt.best_model, map_location='cpu')
    model = best_model['model']

    # 移动到GPU
    model = model.to(opt.device)

    # loss function
    criterion = nn.CrossEntropyLoss().to(opt.device)

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, 'test'),
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers = opt.workers if opt.is_Linux else 0,
        pin_memory=True)
    
    # test
    testing(test_loader, model, criterion, opt.print_freq, opt.device)
