import random
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
from os import listdir

def collate_fn(batch):
    input_batch, output_batch = [], []
    for item in batch:
        input_batch.append(item[0])
        output_batch.append(item[1])
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=0)
    output_batch = torch.Tensor(output_batch)

    return input_batch, output_batch

class IterationDataset(Dataset):
    def __init__(self, sample_list) -> None:
        super().__init__()
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_list[index][0], self.sample_list[index][1]

class IMDBDataset:
    def __init__(self, min_freq=10, max_length=800, path_root='./data/aclImdb') -> None:
        super().__init__()
        self.tokenizer = get_tokenizer('basic_english')
        self.path_root = path_root
        self.min_freq = min_freq
        self.max_length = max_length
        self.load_data()
        self.data_shuffle()
        self.build_vocab(self.min_freq)
        self.create_text_feature()
        self.train_dataset = IterationDataset(self.train_list)
        self.test_dataset = IterationDataset(self.test_list)

    def load_data(self):
        self.train_list, self.test_list = [], []
        stop_words = set(stopwords.words('english'))
        train_pos_filenames = listdir(self.path_root + '/train/pos/')
        train_neg_filenames = listdir(self.path_root + '/train/neg/')
        test_pos_filenames = listdir(self.path_root + '/test/pos/')
        test_neg_filenames = listdir(self.path_root + '/test/neg/')

        max_length = 0

        with tqdm(total=len(train_pos_filenames) + len(train_neg_filenames), desc="加载训练数据") as pbar:
            for filename in train_pos_filenames:
                pbar.update(1)
                with open(self.path_root + '/train/pos/' + filename, 'r') as f:
                    sentence = self.tokenizer(f.readline())
                    text = []
                    for word in sentence:
                        if word.isalpha() and word not in stop_words:
                            text.append(word)
                    text = text[:self.max_length]
                    max_length = max(max_length, len(text))
                    self.train_list.append([text, 1])

            for filename in train_neg_filenames:
                pbar.update(1)
                with open(self.path_root + '/train/neg/' + filename, 'r') as f:
                    sentence = self.tokenizer(f.readline())
                    text = []
                    for word in sentence:
                        if word.isalpha() and word not in stop_words:
                            text.append(word)
                    text = text[:self.max_length]
                    max_length = max(max_length, len(text))
                    self.train_list.append([text, 0])
            
        with tqdm(total=len(test_pos_filenames) + len(test_neg_filenames), desc="加载测试数据") as pbar:
            for filename in test_pos_filenames:
                pbar.update(1)
                with open(self.path_root + '/test/pos/' + filename, 'r') as f:
                    sentence = self.tokenizer(f.readline())
                    text = []
                    for word in sentence:
                        if word.isalpha() and word not in stop_words:
                            text.append(word)
                    text = text[:self.max_length]
                    max_length = max(max_length, len(text))
                    self.test_list.append([text, 1])

            for filename in test_neg_filenames:
                pbar.update(1)
                with open(self.path_root + '/test/neg/' + filename, 'r') as f:
                    sentence = self.tokenizer(f.readline())
                    text = []
                    for word in sentence:
                        if word.isalpha() and word not in stop_words:
                            text.append(word)
                    text = text[:self.max_length]
                    max_length = max(max_length, len(text))
                    self.test_list.append([text, 0])
                    
        print("最大长度:{:d}".format(max_length))

    def create_text_feature(self):
        train_list, test_list = [], []
        for sample in self.train_list:
            feature = []
            for token in sample[0]:
                if token in self.word_to_index:
                    feature.append(self.word_to_index[token])
                else:
                    feature.append(1)
            train_list.append([torch.LongTensor(feature), sample[1]])

        for sample in self.test_list:
            feature = []
            for token in sample[0]:
                if token in self.word_to_index:
                    feature.append(self.word_to_index[token])
                else:
                    feature.append(1)
            test_list.append([torch.LongTensor(feature), sample[1]])

        self.train_list = train_list
        self.test_list = test_list

    def data_shuffle(self):
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)

    def build_vocab(self, min_freq):
        # 只使用训练数据构建词表
        self.word_to_index = {'<pad>':0, '<unk>': 1}
        self.index_to_word = {0:'<pad>', 1: '<unk>'}
        token_freq = {}
        for sample in self.train_list:
            line = sample[0]
            for token in line:
                if token not in token_freq:
                    token_freq[token] = 1
                else:
                    token_freq[token] += 1
        
        # 过滤低频词
        for token in token_freq:
            if token_freq[token] >= min_freq:
                self.word_to_index[token] = len(self.word_to_index)
                self.index_to_word[self.word_to_index[token]] = token

        print("词表长度: {:d}".format(len(self.word_to_index)))
        self.vocab_size = len(self.word_to_index)
        a = 0
        for key in self.word_to_index:
            print(key, self.word_to_index[key])
            if a > 4:
                break
            a += 1

def textclassification(epoch_num=20, batch_size=64, seed=2019):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MyModel(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(in_features=hidden_size*2, out_features=1)
            self.dropout = nn.Dropout(0.5)
            self.sigmoid = nn.Sigmoid()
            self.bn1 = nn.BatchNorm1d(num_features=hidden_size*2)
            # self.bn2 = nn.BatchNorm1d(num_features=1)
            self.initiate_weights()
        
        def initiate_weights(self):
            for name, param in self.lstm.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
            for name, param in self.linear.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
        
        def forward(self, x):
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            x = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
            # x = self.bn1(x)
            x = self.dropout(x)
            x = self.linear(x)
            x = x.squeeze(dim=1)
            output = self.sigmoid(x)

            return output

    dataset = IMDBDataset()
    trainDataloader = DataLoader(dataset=dataset.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    evalDataloader = DataLoader(dataset=dataset.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MyModel(vocab_size=dataset.vocab_size, embedding_size=128, hidden_size=128)
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)

    model.to(device)

    best_result = 0
    min_eval_loss = 10000

    for epoch in range(epoch_num):
        print("Begin Training...")
        model.train()
        total_loss = 0
        # for train_input, train_label in tqdm(trainDataloader, desc="Epoch {:d}".format(epoch + 1)):
        for train_input, train_label in tqdm(trainDataloader, desc="Epoch {:d}".format(epoch + 1)):
            train_input = train_input.to(device)
            train_label = train_label.to(device)
            logits = model(train_input)
            loss = criterion(logits, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch:{:d}; Loss:{:f}".format(epoch + 1, total_loss / len(trainDataloader)))

        # eval
        model.eval()
        eval_loss = 0
        total_acc = 0
        print("Begin Testing...")
        for eval_input, eval_label in tqdm(evalDataloader, desc="Eval"):
            with torch.no_grad():
                eval_input = eval_input.to(device)
                eval_label = eval_label.to(device)
                logits = model(eval_input)
                loss = criterion(logits, eval_label)
                predict = (logits > 0.5)
                total_acc += (predict == eval_label).float().mean()
                eval_loss += loss

        best_result = max(best_result, total_acc / len(evalDataloader))
        min_eval_loss = min(min_eval_loss, eval_loss / len(evalDataloader))

        print("Epoch:{:d} Eval Loss:{:f} Arerage accuracy:{:f}".format(epoch + 1, eval_loss / len(evalDataloader), total_acc / len(evalDataloader)))

    print("Training is over, the best result is {:f}, the min eval loss is {:f}".format(best_result, min_eval_loss))

if __name__ == '__main__':
    textclassification()