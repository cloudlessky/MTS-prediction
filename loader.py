from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import numpy as np
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,2,9'
# print('torch.cuda.device_count()',torch.cuda.device_count())


class MilanDataset(Dataset):
    def __init__(self, data, window, future):
        self.data = data
        self.window = window
        self.future = future

    def __getitem__(self, index):
        if len(self.data.shape) == 1:
            x = self.data[int(index):int(self.window + index)]
            y = self.data[int(index + self.window):int(self.window + index + self.future)]
            return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        else:
            x = self.data[:, int(index):int(self.window + index)]
            y = self.data[:, int(index + self.window):int(self.window + index + self.future)]
            return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        if len(self.data.shape) == 1:
            return len(self.data) - self.window - self.future
        else:
            return self.data.shape[1] - self.window - self.future

def GenerateLoader(set_num, seq, future, batch_size):
    print("starting to load data...")
    # training_set = pd.read_csv('airline-passengers.csv')
    # training_set = pd.read_csv('shampoo.csv')
    # training_set = training_set.iloc[:, 1:2].values
    # training_set = np.load('./data/dataAndCode/region/milan_traffic_region_0.npy')
    if set_num == 1:
        custom_dataset = np.load('custom_dataset.npy')
        sc = MinMaxScaler(feature_range=(-1, 1)) # Preprocess
        custom_dataset = sc.fit_transform(custom_dataset.T).T
    elif set_num == 2:
        sc = StandardScaler()
        with open('data/2016PassengerCount.txt', 'r') as f:
            lines = f.readlines()
            custom_dataset_list = []
            for line in lines[1::2]:
                subdata_str=line.split()
                subdata=[]
                for num_str in subdata_str:
                    num = int(num_str)
                    subdata.append(num)
                custom_dataset_list.append(subdata)
        custom_dataset = np.array(custom_dataset_list)
        custom_dataset = sc.fit_transform(custom_dataset.T).T
    # elif set_num == 3:
    #     # custom_dataset = np.load('../custom_dataset_10.npy')
    #     custom_dataset = np.load('./data/milan_traffic_region.npy')  # [230400 100]
    #     custom_dataset = custom_dataset.T  # [100 230400]
    #     custom_dataset = custom_dataset.reshape(10000, -1)  # [10000 230400]
    #     sc = MinMaxScaler(feature_range=(-1, 1))
    #     custom_dataset = sc.fit_transform(custom_dataset.T).T
    #     # custom_dataset = sc.fit_transform(custom_dataset.T).T
    elif set_num == 3:
        # custom_dataset = np.load('./data/milan_traffic_region.npy')
        # custom_dataset = custom_dataset.T  # [100 230400]
        # custom_dataset = custom_dataset.reshape(10000, -1)  # [10000 230400]
        # custom_dataset = custom_dataset[:1000,:]
        # sc = MinMaxScaler(feature_range=(-1, 1))
        # custom_dataset = sc.fit_transform(custom_dataset.T)
        for tt in range(0, 100):
            # print('***********tt*************',tt)
            # *********load data************
            custom_dataset_tt = np.load('./data/milan_traffic_region_{}.npy'.format(list[tt]))  # 1000个任务 [4320]
            training_data_signal = custom_dataset_tt.reshape(-1, 1)  # [4320 1]
            training_data_signal = torch.from_numpy(training_data_signal)#numpy-tensor
            #print('training_data_signal.shape',training_data_signal.shape)
            if tt==0:
                custom_dataset = training_data_signal
            else:
                custom_dataset = torch.cat((custom_dataset,training_data_signal),1)
        custom_dataset = custom_dataset.numpy()
        sc = MinMaxScaler(feature_range=(-1, 1))
        custom_dataset = sc.fit_transform(custom_dataset.T).T
        # custom_dataset = sc.fit_transform(custom_dataset.T)
        # print('IN loader,custom_dataset.shape', custom_dataset.shape)  # 1000 4320
        # ## 创建二维数组
        # #b = np.array([1, 2, 3], [4, 5, 6])
        # # 将二维数组写入 CSV
        # np.savetxt( "dataset1.csv", b, delimiter="," )


    elif set_num == 4:
        custom_dataset = np.load('custom_dataset_1000.npy')
        sc = MinMaxScaler(feature_range=(-1, 1))
        custom_dataset = sc.fit_transform(custom_dataset.T).T
    elif set_num == 5:
        custom_dataset = np.load('custom_dataset_50.npy')
        sc = MinMaxScaler(feature_range=(-1, 1))
        custom_dataset = sc.fit_transform(custom_dataset.T).T

    custom_dataset = custom_dataset.T
    [num_tasks, seq_len] = custom_dataset.shape
    print('IN loader,custom_dataset.shape',custom_dataset.shape)#1000 4320

    #*****划分数据集长度******
    train_set = custom_dataset[:, :int(seq_len * 0.5)]
    valid_set = custom_dataset[:, int(seq_len * 0.5):int(seq_len * 0.7)]
    test_set = custom_dataset[:, int(seq_len * 0.7):]
    print('train_set,valid_set,test_set.shape',train_set.shape,valid_set.shape,test_set.shape)
    #(1000, 2160) (1000, 864) (1000, 1296)
    #划分窗口
    milan_dataset = MilanDataset(train_set, seq, future)
    train_loader = DataLoader(milan_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    milan_dataset = MilanDataset(valid_set, seq, future)
    valid_loader = DataLoader(milan_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    milan_dataset = MilanDataset(test_set, seq, future)
    test_loader = DataLoader(milan_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader

# def sigmoid(x):
#
#     return 1/(1+np.exp(-x))

if __name__ == '__main__':
    train_loader, valid_loader, test_loader = GenerateLoader(3, 60, 6, 32)
    #print('len(test_loader.dataset)',len(test_loader.dataset))#68
    # w=0.8
    # bias=-5
    # for t in range(0,13):
    #     decay = 1 - sigmoid(w * t + bias)  # t=0时，decay~1
    #     print('t',t,'decay', decay)
    for input, target in test_loader:
        print(input.size(), target.size())#torch.Size([32,1000, 60]) torch.Size([32,1000, 6]) [batch_size num_of_tasks seq_len]
        break