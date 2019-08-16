'''
@Author: Gordon Lee
@Date: 2019-08-11 19:12:09
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-13 16:38:22
@Description: 
'''
import os
import json
import torch
import pandas as pd
import numpy as np
import warnings
from utils import load_embeddings
from collections import Counter
from config import Config

warnings.filterwarnings("ignore") # 忽略输出警告

def create_input_files(data_name, SST_path, emb_file, emb_format, output_folder, min_word_freq, max_len):
    '''
    对数据集进行预处理
    
    :param data_name: SST-1/SST-2
    :param SST_path: Stanford Sentiment Treebank数据集的路径
    :param emb_file: 预训练词向量文件路径
    :param emb_format: 词向量格式 glove or word2vec
    :param output_folder: 处理后的数据集保存路径
    :param min_word_freq: 最小词频
    :param max_len: 最大采样长度
    '''

    # Sanity check
    assert data_name in {'SST-1', 'SST-2'}
    
    
    # 读入数据集
    print('Preprocess datasets...')
    datasetSentences = pd.read_csv(SST_path + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(SST_path + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(SST_path + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(SST_path + 'sentiment_labels.txt', sep='|')  

    # 将多个表进行内连接合并
    dataset = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary),sentiment_labels)
    

    def labeling(data_name, sentiment_value):
        '''
        将情感值转为标签
        
        :param data_name: SST-1/SST-2
        :param sentiment_value: sentiment_value
        :return: label
        '''
        if data_name == 'SST-1':
            if sentiment_value <= 0.2:
                return 0 # very negative
            elif sentiment_value <= 0.4:
                return 1 # negative
            elif sentiment_value <= 0.6:
                return 2 # neutral
            elif sentiment_value <= 0.8:
                return 3 # positive
            elif sentiment_value <= 1:
                return 4 # very positive
        else:
            if sentiment_value <= 0.4:
                return 0 # negative
            elif sentiment_value > 0.6:
                return 1 # positive
            else:
                return -1 # drop neutral  
            
    # 将情感值转为标签
    dataset['sentiment_label'] = dataset['sentiment values'].apply(lambda x: labeling(data_name, x))
    dataset = dataset[dataset['sentiment_label'] != -1]
    

    
    def check_not_punctuation(token):
        '''
        检查token是否完全由非数字字母字符组成，比如``
        
        :param s: sentence
        :return: bool
        '''
        for ch in token:
            if ch.isalnum(): return True
        return False
    
    def filter_punctuation(s):
        '''
        将句子转为小写，同时过滤标点符号等
        
        :param s: sentence
        :return: token list
        '''
        s = s.lower().split(' ')
        return [token for token in s if check_not_punctuation(token)]
    
    # 对句子进行预处理
    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))
    
    
    
    # 创建词表
    word_freq = Counter()
    valid_idx = []
    for i,tokens in enumerate(dataset['sentence']):
        word_freq.update(tokens)
        if len(tokens) <= max_len: # 采样长度不超过max_len
            valid_idx.append(i)
    dataset = dataset.iloc[valid_idx, :]

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}  
    word_map['<unk>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    
    

    def tokens_to_idx(tokens):
        '''
        将token转为索引
        
        :param tokens: token list
        :return: index list
        '''
        return [word_map.get(word, word_map['<unk>']) for word in tokens] + [word_map['<pad>']] * (max_len - len(tokens))   
    
    # 将token转成索引
    dataset['token_idx'] = dataset['sentence'].apply(lambda x: tokens_to_idx(x))
    

    
    # 加载并保存预训练词向量
    pretrain_embed, embed_dim = load_embeddings(emb_file,  emb_format, word_map)
    embed = dict()
    embed['pretrain'] = pretrain_embed
    embed['dim'] = embed_dim
    torch.save(embed, output_folder + data_name + '_' + 'pretrain_embed.pth')
    
    
    # 保存word_map
    with open(os.path.join(output_folder, data_name + '_' + 'wordmap.json'), 'w') as j:
            json.dump(word_map, j)
    
            
    # 保存处理好的数据集
    # train
    dataset[dataset['splitset_label']==1][['token_idx','sentiment_label']].to_csv(output_folder + data_name + '_' + 'train.csv',index=False)
    # test
    dataset[dataset['splitset_label']==2][['token_idx','sentiment_label']].to_csv(output_folder + data_name + '_' + 'test.csv',index=False)
    # dev
    dataset[dataset['splitset_label']==3][['token_idx','sentiment_label']].to_csv(output_folder + data_name + '_' + 'dev.csv',index=False)
    
    print('Preprocess End\n')
    
    

if __name__ == "__main__":
    opt = Config()
    create_input_files(data_name=opt.data_name,
                       SST_path=opt.SST_path,
                       emb_file=opt.emb_file,
                       emb_format=opt.emb_format,
                       output_folder=opt.output_folder, 
                       min_word_freq=opt.min_word_freq, 
                       max_len=opt.max_len)