from torchvision import models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from utils.data_set import HymenopteraDataset, ImageTransform, make_datapath_list

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

#最後の層をout=2に変更
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train()

print('setting network is done: loading weight and set train mode.')

criterion = nn.CrossEntropyLoss()

# 学習させるパラメータを格納
params_to_update = []

update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        #print(name)
    else:
        param.requires_grad = False

#print("--------------")
#print(params_to_update)

optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if(epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) # ラベルを予測

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    #total of loss
                    epoch_loss += loss.item() * inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


def main():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    num_epochs=2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
