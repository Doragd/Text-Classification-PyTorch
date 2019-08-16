'''
@Author: Gordon Lee
@Date: 2019-08-09 14:22:50
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-13 16:40:09
@Description: 
'''
import torch
import pandas as pd
from torch.utils.data import Dataset

class SSTreebankDataset(Dataset):
    '''
    创建dataloader
    '''
    
    def __init__(self, data_name, output_folder, split):
        '''
        :param output_folder: 数据文件所在路径
        :param split: 'train', 'dev', or 'test'
        '''
        self.split = split
        assert self.split in {'train', 'dev', 'test'}
        
        self.dataset = pd.read_csv(output_folder + data_name + '_' + split + '.csv')

        self.dataset_size = len(self.dataset)
        
    def __getitem__(self, i):

        sentence = torch.LongTensor(eval(self.dataset.iloc[i]['token_idx'])) # sentence shape [max_len]
        sentence_label = self.dataset.iloc[i]['sentiment_label']
        
        return sentence, sentence_label

    def __len__(self):
        
        return self.dataset_size