'''
@Author: Gordon Lee
@Date: 2019-08-12 21:53:17
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-13 17:58:30
@Description: 
'''

class Config(object):
    '''
    全局配置参数
    '''
    status = 'train' # 执行 train_eval or test, 默认执行train_eval
    use_model = 'TextCNN' # 使用何种模型, 默认使用TextCNN
    output_folder = 'output_data/'  # 已处理的数据所在文件夹
    data_name = 'SST-2' # SST-1(fine-grained) SST-2(binary)
    SST_path  = 'data/stanfordSentimentTreebank/' # 数据集所在路径
    emb_file = 'data/glove.6B.300d.txt' # 预训练词向量所在路径
    emb_format = 'glove' # embedding format: word2vec/glove
    min_word_freq = 1 # 最小词频
    max_len = 40 # 采样最大长度