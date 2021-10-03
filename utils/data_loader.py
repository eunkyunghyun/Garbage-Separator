from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import torch


class NkDataSet(Dataset):
    def __init__(self, file_path):
        self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(file_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0][1:])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1][1:], dtype=np.float32)
        self.label_arr = torch.from_numpy(self.label_arr)
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        img_name = self.image_arr[index - 1]
        img_as_img = Image.open(img_name)
        img_as_tensor = self.trans(img_as_img)
        img_label = self.label_arr[index - 1]
        return img_as_tensor, img_label, img_name

    def __len__(self):
        return self.data_len


def get_data_loader(args):
    csv_path = "./labels/train_label.csv"
    custom_dataset = NkDataSet(csv_path)

    test_csv_path = "./labels/test_label.csv"
    test_custom_dataset = NkDataSet(test_csv_path)

    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)
    # [batch_size] train: 100, test: 1
    # [shuffle] train: True, test: False

    test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=1, shuffle=False)
    # [batch_size] train: 1, test: 1
    # [shuffle] train: False, test: False

    print("data set length = %d" % len(test_loader))

    return train_loader, test_loader
