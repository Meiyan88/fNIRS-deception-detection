import os
from sklearn.model_selection import KFold
import shutil
import numpy as np
import scipy.io as sio
import torch

def Load_Dataset_f(data_path):
    """
    Load Dataset f

    Args:
        data_path (str): dataset path.

    Returns:
        feature : fNIRS signal data.
        label : fNIRS labels
        BASset: personality

    """
    BASset = []
    HbRset = []
    label = []
    path = data_path
    scale_path = os.path.join(path,'scale')
    pathdir = os.listdir(path)
    for pathlist in pathdir:
       if pathlist == 'scale':
           continue
       feature_HbR = sio.loadmat(path + pathlist)["timepoint_hbo"]
       feature_HbR = np.array(feature_HbR)
       HbRset.append(feature_HbR)

       Label = float(pathlist[-5])
       label.append(Label)

       num = pathlist.split('.')[0]
       num = num[-11:]
       sheet = sio.loadmat(os.path.join(scale_path, 'scale'+num+'.mat'))["Scale"]
       sheet = np.array(sheet)
       BASset.append(sheet)


    dataset = np.array(HbRset).transpose((0, 3, 2, 1))
    BASset = np.array(BASset)
    label = np.array(label)


    print('feature ', dataset.shape)
    print('label ', label.shape)

    return dataset, label, BASset


def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def split_5_fold_data():
    random_state = 888
    in_dir = "/public/envs/Time/"
    in_scale_dir = "/public/envs/scale2/"
    files = os.listdir(in_dir)
    print(files)
    files = os.listdir(in_dir)
    print(files)
    save_path = "/public/envs/Time_fold/five_fold_" + str(random_state)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    total_num = len(files)

    index = np.zeros(total_num)
    skf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_ind = 0
    for temp_train_ind, temp_test_ind in skf.split(index):
        fold_ind += 1
        print(temp_test_ind)

        train_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "train")
        scale_train_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "train","scale")
        test_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "test")
        scale_test_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "test","scale")
        val_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "val")
        scale_val_data_path = os.path.join(save_path, 'fold' + str(fold_ind), "val","scale")
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
        if not os.path.exists(scale_train_data_path):
            os.makedirs(scale_train_data_path)
        if not os.path.exists(scale_test_data_path):
            os.makedirs(scale_test_data_path)
        if not os.path.exists(scale_val_data_path):
            os.makedirs(scale_val_data_path)
        if not os.path.exists(scale_val_data_path):
            os.makedirs(scale_val_data_path)


        for train_index in temp_train_ind:
            for each_mat in os.listdir(os.path.join(in_dir,files[train_index])):
                each_mat_path = os.path.join(in_dir, files[train_index],each_mat)
                shutil.copy(each_mat_path, train_data_path)
                num = each_mat.split('.')[0]
                num = num[-11:]
                scale_mat_path = os.path.join(in_scale_dir, 'scale'+num+'.mat')
                shutil.copy(scale_mat_path, scale_train_data_path)

        i = 0
        for test_index in temp_test_ind:
            i = i+1
            if i > 6:
                for each_mat in os.listdir(os.path.join(in_dir,files[test_index])):
                    each_mat_path = os.path.join(in_dir, files[test_index],each_mat)
                    shutil.copy(each_mat_path, test_data_path)
                    num1 = each_mat.split('.')[0]
                    num1 = num1[-11:]
                    scale_mat_path = os.path.join(in_scale_dir, 'scale'+num1+'.mat')
                    shutil.copy(scale_mat_path, scale_test_data_path)
            else:
                for each_mat in os.listdir(os.path.join(in_dir,files[test_index])):
                    each_mat_path = os.path.join(in_dir, files[test_index],each_mat)
                    shutil.copy(each_mat_path, val_data_path)
                    num1 = each_mat.split('.')[0]
                    num1 = num1[-11:]
                    scale_mat_path = os.path.join(in_scale_dir, 'scale'+num1+'.mat')
                    shutil.copy(scale_mat_path, scale_val_data_path)


class Dataset(torch.utils.data.Dataset):
    """
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        sheet: personality assessments for input data
        transform: Z-score normalization is used to accelerate convergence (default:True).
    """
    def __init__(self, feature, label,  SHEET, transform=True):
        self.feature = feature
        self.label = label
        self.sheet = SHEET
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.sheet = torch.tensor(self.sheet, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item], self.sheet[item]
        # ,