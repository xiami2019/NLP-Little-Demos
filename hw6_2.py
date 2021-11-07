'''
train_data = torch.randn((5,6,10))
train_label = torch.ones((5,1))
test_data = torch.randn(3,6,10)
epoch_num = 5
ret = torch.rand((3,))
'''
import torch
import torch.nn as nn

def textclassification(train_data, train_label, test_data, epoch_num=10):
    class MyModel(nn.Module):
        def __init__(self, embedding_size=10, hidden_size=10):
            super().__init__()
            self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
            self.linear = nn.Linear(in_features=hidden_size*2, out_features=1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x, _ = self.gru(x)
            x = x[:,-1,:]
            x = self.linear(x)
            output = self.sigmoid(x)

            return output

    model = MyModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.1)

    for epoch in range(epoch_num):
        logits = model(train_data)
        loss = criterion(logits, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predict = torch.argmax(model(test_data), -1)

    return predict

if __name__ == '__main__':
    train_data = torch.randn((5,6,10))
    train_label = torch.ones((5,1))
    test_data = torch.randn(3,6,10)
    epoch_num = 5
    ret = textclassification(train_data, train_label, test_data, epoch_num)
    print(ret)