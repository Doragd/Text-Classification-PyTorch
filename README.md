# Text-Classification-PyTorch :whale2:

Here is a new boy :bow:  who wants to become a NLPer and his repository for Text Classification.  Besides TextCNN and TextAttnBiLSTM, more models will be added in the near future. 

Thanks for you Star:star:, Fork and Watch！

## Dataset

* [Stanford Sentiment Treebank(SST)](nlp.stanford.edu/sentiment/code.html)
  * SST-1: 5 classes(fine-grained),  SST-2: 2 classes(binary)
* Preprocess
  * Map sentiment values to labels
  * Remove tokens consisting of all non-alphanumeric characters, such as `...`

## Pre-trained Word Vectors

* [Word2Vec](https://code.google.com/archive/p/word2vec/) : `GoogleNews-vectors-negative300.bin`
* [GloVe](https://nlp.stanford.edu/projects/glove/) : `glove.840B.300d.txt`
  * Because the OOV Rate of *GloVe* is lower than *Word2Vec* and the experiment performance is also better than the other one, we use *GloVe* as pre-trained word vectors.
  * Options for different format word vectors are still preserved in the code.

## Model

* TextCNN
  
  * Paper: [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181)
  * See：`models/TextCNN.py`
  
  ![](https://ws1.sinaimg.cn/large/72cf269fly1g6229o5a47j20m609c74t.jpg)
  
* TextAttnBiLSTM
  
  * Paper: [Attention-Based Bidirection LSTM for Text Classification](https://www.aclweb.org/anthology/P16-2034)
  * See: `models/TextAttnBiLSTM.py`

![](https://ws1.sinaimg.cn/large/72cf269fly1g622af7rxij20la0axq3g.jpg)

## Result

* Baseline from the paper

| model            | SST-1    | SST-2    |
| ---------------- | -------- | -------- |
| CNN-rand         | 45.0     | 82.7     |
| CNN-static       | 45.5     | 86.8     |
| CNN-non-static   | **48.0** | 87.2     |
| CNN-multichannel | 47.4     | **88.1** |

* Re-Implementation

| model              | SST-1      | SST-2      |
| ------------------ | ---------- | ---------- |
| CNN-rand           | 34.841     | 74.500     |
| CNN-static         | 45.056     | 84.125     |
| CNN-non-static     | 46.974     | 85.886     |
| CNN-multichannel   | 45.129     | **85.993** |
| Attention + BiLSTM | 47.015     | 85.632     |
| Attention + BiGRU  | **47.854** | 85.102     |

## Requirement

Please install the following library requirements first.

```markdown
pandas==0.24.2
torch==1.1.0
fire==0.1.3
numpy==1.16.2
gensim==3.7.3
```

## Structure

```python
│  .gitignore
│  config.py            # Global Configuration
│  datasets.py          # Create Dataloader
│  main.py 
│  preprocess.py
│  README.md
│  requirements.txt
│  utils.py   
│  
├─checkpoints           # Save checkpoint and best model
│      
├─data                  # pretrained word vectors and datasets
│  │  glove.6B.300d.txt
│  │  GoogleNews-vectors-negative300.bin
│  └─stanfordSentimentTreebank # datasets folder
│          
├─models
│      TextAttnBiLSTM.py
│      TextCNN.py
│      __init__.py
│      
└─output_data           # Preprocessed data and vocabulary, etc.
```

## Usage

* Set global configuration parameters in config.py

* Preprocess the datasets 

```shell
$python preprocess.py
```

* Train

```shell
$python main.py run
```

You can set the parameters in the `config.py` and `models/TextCNN.py` or `models/TextAttnBiLSTM.py` in the command line.

```shell
$python main.py run [--option=VALUE]
```

For example，

```shell
$python main.py run --status='train' --use_model="TextAttnBiLSTM"
```

* Test

```shell
$python main.py run --status='test' --best_model="checkpoints/BEST_checkpoint_SST-2_TextCNN.pth"
```

## Conclusion

* The `TextCNN` model uses the n-gram-like convolution kernel extraction feature, while the `TextAttnBiLSTM` model uses BiLSTM to capture semantics and long-term dependencies, combined with the attention mechanism for classification.
* TextCNN Parameter tuning:
  * glove is better than word2vec
  * Use a smaller batch size
  * Add weight decay ($l_2$ constraint), learning rate decay, early stop, etc.
  * Do not set `padding_idx=0` in embedding layer
* TextAttnBiLSTM
  * Apply dropout on embedding layer, LSTM layer, and fully-connected layer

## Acknowledge

* Motivated by https://github.com/TobiasLee/Text-Classification
* Thanks to https://github.com/bigboNed3/chinese_text_cnn
* Thanks to https://github.com/ShawnyXiao/TextClassification-Keras

## Reference

[1] [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)

[2] [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820)

[3] [Attention-Based Bidirection LSTM for Text Classification](https://www.aclweb.org/anthology/P16-2034)

