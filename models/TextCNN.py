'''
@Author: Gordon Lee
@Date: 2019-08-09 16:29:55
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-16 19:00:19
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

class ModelConfig(object):
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
    batch_size = 32 # batch_size
    workers = 4  # 多处理器加载数据
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5 # 权重衰减率
    decay_epoch = 20 # 多少个epoch后执行学习率衰减
    improvement_epoch = 6 # 多少个epoch后执行early stopping
    is_Linux = True # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint = None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextCNN' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    kernel_num = 100 # kernel数量
    kernel_sizes = [3,4,5] # 不同尺寸的kernel
    dropout = 0.5 # dropout
    embed_dim = 128 # 未使用预训练词向量的默认值
    static = True # 是否使用预训练词向量, static=True, 表示使用预训练词向量
    non_static = True # 是否微调，non_static=True,表示微调
    multichannel = True # 是否多通道


class ModelCNN(nn.Module):
    '''
    TextCNN: CNN-rand, CNN-static, CNN-non-static, CNN-multichannel
    '''
    def __init__(self, vocab_size, embed_dim, kernel_num, kernel_sizes, class_num, pretrain_embed, dropout, static, non_static, multichannel):
        '''
        :param vocab_size: 词表大小
        :param embed_dim: 词向量维度
        :param kernel_num: kernel数目
        :param kernel_sizes: 不同kernel size
        :param class_num: 类别数
        :param pretrain_embed: 预训练词向量
        :param dropout: dropout
        :param static: 是否使用预训练词向量, static=True, 表示使用预训练词向量
        :param non_static: 是否微调，non_static=True,表示不微调
        :param multichannel: 是否多通道
        '''
        super(ModelCNN, self).__init__()
        
        # 初始化为单通道
        channel_num = 1
        
        # 随机初始化词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 使用预训练词向量
        if static:
            self.embedding = self.embedding.from_pretrained(pretrain_embed, freeze=not non_static)
        
        # 微调+固定预训练词向量
        if multichannel:
            # defalut: freeze=True, 即默认embedding2是固定的
            self.embedding2 = nn.Embedding(vocab_size, embed_dim).from_pretrained(pretrain_embed)
            channel_num = 2
        else:
            self.embedding2 = None
    
        # 卷积层, kernel size: (size, embed_dim), output: [(batch_size, kernel_num, h,1)] 
        self.convs = nn.ModuleList([
            nn.Conv2d(channel_num, kernel_num, (size, embed_dim)) 
            for size in kernel_sizes
        ])

        
        # 1维最大池化层,因为无法确定feature map大小，所以放在forward里面
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
    
    def forward(self, x):
        '''
        :params x: (batch_size, max_len)
        :return x: logits
        '''
        
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1) # (batch_size, 2, max_len, word_vec)
        else:
            x = self.embedding(x).unsqueeze(1) # (batch_size, 1, max_len, word_vec)
        
        # 卷积    
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, kernel_num, h)]
        # 池化
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(batch_size, kernel_num)]
        # flatten
        x = torch.cat(x, 1) # (batch_size, kernel_num * len(kernel_sizes)) 
        # dropout
        x = self.dropout(x)
        # fc
        x = self.fc(x) # logits, 没有softmax, (batch_size, class_num)
        
        return x 



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
        if opt.static == False: embed_dim = opt.embed_dim
        model = ModelCNN(vocab_size=len(word_map), 
                      embed_dim=embed_dim, 
                      kernel_num=opt.kernel_num, 
                      kernel_sizes=opt.kernel_sizes, 
                      class_num=opt.class_num,
                      pretrain_embed=pretrain_embed,
                      dropout=opt.dropout, 
                      static=opt.static, 
                      non_static=opt.non_static, 
                      multichannel=opt.multichannel)
    
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
              device=opt.device)
        
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

