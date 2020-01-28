from utils.data_set import ImageTransform, make_datapath_list, HymenopteraDataset

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def main():
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase="train")
    val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase="val")

    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)

    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.train()

    criterion = nn.CrossEntropyLoss()

    #store params in params_to_update for fine tuning

    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)

        else:
            param.requires_grad = False

    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
    ], momentum=0.9)

    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    #ある程度ネットワークが固定であれば高速化
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('----------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                #initialize optimizer
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)#.dataはTensorを返す。.detach()の使用推奨

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    save(net)

def save(net):
    checkpoint_dir = "./checkpoint/"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    save_path = "./checkpoint/weights_fine_tuning.pth"
    torch.save(net.state_dict(), save_path)

def load(net):
    load_path = "./checkpoint/weights_fine_tuning.pth"
    load_weights = torch.load(load_path)#(, map_location={'cuda:0': 'cpu'}) GPU->CPU
    net.load_state_dict(load_weights)

if __name__ == "__main__":
    main()