'''
train_data = torch.randn((3,6,10))
train_label = torch.ones((3,1))
test_data = torch.randn(3,6,10)
epoch_num = 5
ret = torch.rand((3,))
'''
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip

def imageAugmentation(images):
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.module1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
                nn.ReLU()
            )


            self.module2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3),
                nn.ReLU()
            )

            self.module3 = nn.Sequential(
                nn.Linear(in_features=864, out_features=10),
                nn.Softmax(dim=1)
            )
            
        
        def forward(self, x):
            x = self.module1(x)
            x = self.module2(x)
            x = x.reshape(10, -1)
            x = self.module3(x)

            return x

    def augmentation(images):
        # 按照0.5的概率进行翻转
        

        return images

    model = MyModel()
    images = augmentation(images)
    ret = model(images)

    return ret

if __name__ == '__main__':
    images = torch.randn((10,3,16,16))
    ret = imageAugmentation(images=images)
    print(ret.shape)