# %%
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler


### loading data
def load(input_dim, seq_length, future_length, set_num):
    print("starting to load data...")
    # training_set = pd.read_csv('airline-passengers.csv')
    # training_set = pd.read_csv('shampoo.csv')
    # training_set = training_set.iloc[:, 1:2].values
    # training_set = np.load('./data/dataAndCode/region/milan_traffic_region_0.npy')
    if set_num == 1:
        custom_dataset = np.load('./data/milan_traffic_region_0.npy')  # [4320 1]
        sc = MinMaxScaler(feature_range=(-1, 1))
        custom_dataset = torch.from_numpy(custom_dataset)
        custom_dataset = custom_dataset.reshape(-1, 1)
        custom_dataset = sc.fit_transform(custom_dataset.T)
        # custom_dataset = sc.fit_transform(custom_dataset.T).T
    elif set_num == 2:
        sc = StandardScaler()
        with open('data/2016PassengerCount.txt', 'r') as f:
            lines = f.readlines()
            custom_dataset_list = []
            for line in lines[1::2]:
                subdata_str = line.split()
                subdata = []
                for num_str in subdata_str:
                    num = int(num_str)
                    subdata.append(num)
                custom_dataset_list.append(subdata)
        custom_dataset = np.array(custom_dataset_list)
        custom_dataset = sc.fit_transform(custom_dataset.T).T
        print('custom_dataset.shape',custom_dataset.shape)#[268 384]

    def sliding_windows(data, seq_length, future_length):
        x = []
        y = []

        for i in range(len(data) - seq_length - future_length):
            _x = data[i:(i + seq_length)]
            _y = data[i + seq_length: i + seq_length + future_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    if set_num==1:
        numTasks = 100
    elif set_num==2:
        numTasks = custom_dataset.shape[0]

    print('numTasks',numTasks)#100
    print('custom_dataset.shape',custom_dataset.shape)#[1 4320] [268 384]
    training_data = custom_dataset[0, :].reshape(-1, 1)
    x, y = sliding_windows(training_data, seq_length, future_length)
    train_size = int(len(y) * 0.5)
    if set_num==2 and seq_length==80:
        valid_size = int(len(y) * 0.3)
    else:
        valid_size = int(len(y) * 0.2)
    #test_size = len(y) - train_size - valid_size
    test_size = int(len(y) * 0.3)
    print('train_size',train_size,valid_size,test_size)#2127 850 1277

    # dataX = Variable(torch.Tensor(np.array(x)))
    # dataY = Variable(torch.Tensor(np.array(y)))

    # trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    # trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    #
    # testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    # testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    trainX = np.ones((numTasks, train_size, seq_length, input_dim))
    trainY = np.ones((numTasks, train_size, future_length, input_dim))

    validX = np.ones((numTasks, valid_size, seq_length, input_dim))
    validY = np.ones((numTasks, valid_size, future_length, input_dim))

    testX = np.ones((numTasks, test_size, seq_length, input_dim))
    testY = np.ones((numTasks, test_size, future_length, input_dim))

    # trainData_id = []
    # testData_id = []
    List = [764, 73, 812, 396, 135, 629, 167, 727, 249, 698, 107, 346, 24, 517, 484, 585, 379, 342, 68, 169, 592,
                  702, 246, 143, 275, 554, 962, 904, 736, 669, 321, 399, 540, 492, 965, 168, 557, 48, 148, 280, 602,
                  384, 936, 218, 961, 393, 994, 926, 792, 527, 824, 976, 443, 662, 229, 251, 591, 834, 71, 972, 532,
                  217, 555, 851, 237, 801, 621, 224, 631, 128, 853, 620, 868, 628, 82, 863, 525, 846, 359, 967, 470,
                  165, 613, 944, 358, 461, 296, 117, 369, 177, 701, 958, 713, 320, 418, 649, 276, 191, 487, 236]
    #List = [901, 249, 495, 92, 384, 971, 845, 250, 326, 986, 106, 323, 814, 239, 874, 757, 444, 121, 897, 843, 780, 943, 219, 2, 763, 514, 414, 300, 847, 707, 203, 387, 448, 259, 773, 663, 96, 36, 930, 989, 303, 420, 484, 496, 794, 611, 12, 638, 213, 920, 786, 999, 649, 785, 810, 438, 895, 733, 378, 916, 386, 68, 0, 958, 861, 578, 524, 616, 90, 680, 799, 684, 33, 189, 907, 308, 184, 413, 4, 366, 330, 714, 208, 806, 634, 6, 726, 739, 361, 990, 641, 728, 637, 236, 99, 890, 163, 371, 278, 993]
    random.seed(2020)
    for tt in range(0, numTasks):
        if set_num==1:
            training_set = np.load('./data/milan_traffic_region_{}.npy'.format(List[tt]))
            training_data = sc.fit_transform(training_set.reshape(-1, 1))
        elif set_num==2:
            training_data = custom_dataset[tt, :].reshape(-1, 1)
        #print('training_data.shape',training_data.shape)#[4320 1]
        x, y = sliding_windows(training_data, seq_length, future_length)
        temp_shuffle = list(range(0, len(y)))
        # random.shuffle(temp_shuffle)
        trainX[tt, :, :, :] = np.array(x[temp_shuffle[0:train_size]])
        trainY[tt, :, :, :] = np.array(y[temp_shuffle[0:train_size]])

        validX[tt, :, :, :] = np.array(x[temp_shuffle[train_size:train_size + valid_size]])
        validY[tt, :, :, :] = np.array(y[temp_shuffle[train_size:train_size + valid_size]])
        # for ii in range(0, train_size):
        #     trainData_id.append((tt, ii))
        testX[tt, :, :, :] = np.array(x[temp_shuffle[len(x)-test_size:len(x)]])
        testY[tt, :, :, :] = np.array(y[temp_shuffle[len(y)-test_size:len(y)]])
        # testX[tt, :, :, :] = np.array(x[temp_shuffle[train_size + valid_size:len(x)]])
        # testY[tt, :, :, :] = np.array(y[temp_shuffle[train_size + valid_size:len(y)]])

        # for ii in range(train_size, len(x)):
        #     testData_id.append((tt, ii))

    print(trainX.shape, trainY.shape, testX.shape, testY.shape, validX.shape, validY.shape)
    #(1000, 2127, 60, 1) (1000, 2127, 6, 1) (1000, 1277, 60, 1) (1000, 1277, 6, 1) (1000, 850, 60, 1) (1000, 850, 6, 1)
    #(100, 2127, 60, 1) (100, 2127, 6, 1) (100, 1277, 60, 1) (100, 1277, 6, 1) (100, 850, 60, 1) (100, 850, 6, 1)
    #(268, 168, 40, 1) (268, 168, 8, 1) (268, 101, 40, 1) (268, 101, 8, 1) (268, 67, 40, 1) (268, 67, 8, 1)
    #(268, 168, 40, 1) (268, 168, 8, 1) (268, 100, 40, 1) (268, 100, 8, 1) (268, 100, 40, 1) (268, 100, 8, 1)
    trainX = Variable(torch.Tensor(trainX)).float()
    trainY = Variable(torch.Tensor(trainY)).float()

    validX = Variable(torch.Tensor(validX)).float()
    validY = Variable(torch.Tensor(validY)).float()

    testX = Variable(torch.Tensor(testX)).float()
    testY = Variable(torch.Tensor(testY)).float()

    print("loading data finished...")
    return trainX, trainY, validX, validY, testX, testY
    # return trainX, trainY, testX, testY, trainData_id, testData_id
    # return dataX, dataY, trainX, trainY, testX, testY


# def get_batches(x, y, batch_size, trainData_id, testData_id):
def get_batches(x, y, batch_size):
    """
    Generate inputs and targets in a batch-wise fashion for feed-dict
    Args:
        x: entire source sequence array
        y: entire output sequence array
        batch_size: batch size
    """
    epoch_completed = False
    # print(x.shape[0])
    # y_batch_id = []
    for batch_i in range(0, x.shape[1] // batch_size):#2127/32 batch_num
        task_completed = False
        for task_id in range(0, x.shape[0]):#1000
            # for task_id in range(0, x.shape[0]):
            #     for batch_i in range(0, x.shape[1] // batch_size):
            x_batch_id = []
            start_i = batch_i * batch_size
            #print('x.shape', x.shape)#([1000, 850, 60, 1])
            x_batch = x[task_id][start_i: start_i + batch_size]
            y_batch = y[task_id][start_i: start_i + batch_size]

            for ins_id in range(batch_size):
                x_batch_id.append((task_id, batch_i * batch_size + ins_id))
                # y_batch_id.append(task_id, batch_i*batch_size+ins_id)
            # x_batch_id = trainData_id[start_i : start_i + batch_size]
            # y_batch_id = testData_id[start_i : start_i + batch_size]
            #print('###x_batch.shape', x_batch.shape)#([32, 60, 1])
            # x_batch = torch.from_numpy(x_batch)
            x_batch = x_batch.permute(1, 0, 2)
            #print('x_batch.shape',x_batch.shape)#([60, 32, 1])

            # y_batch = torch.from_numpy(y_batch)
            y_batch = y_batch.permute(1, 0, 2)
            # source_sentence_length = [np.count_nonzero(seq) for seq in x_batch]
            # target_sentence_length = [np.count_nonzero(seq) for seq in y_batch]
            if task_id == x.shape[0] - 1:
                task_completed = True
            if task_id == x.shape[0] - 1 and batch_i == x.shape[1] // batch_size - 1:
                epoch_completed = True

            yield x_batch, y_batch, epoch_completed, x_batch_id


# %%
if __name__ == '__main__':
    # trainX, trainY, validX, validY, testX, testY = load(1, 60, 6, 1)
    trainX, trainY, validX, validY, testX, testY = load(1, 40, 8, 2)
    # print(trainY[])
    # print(trainX.shape)
    # print(dataY)
    # print(trainX)
    # print(np.shape(trainX))
    # print(trainY)
    # print(np.shape(trainY))
    # print(testX)
    # print(np.shape(testX))
    # print(testY)
    # print(np.shape(testY))
    # a = torch.randn(10)
    # print(a.view(1, 1, 10))
    for total_id, (input_batch, output_batch, epoch_completed, x_batch_id) in \
            enumerate(get_batches(validX, validY, 32)):
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        print(input_batch.shape)##([60, 32, 1])  ([40, 32, 1])
        #     print(x_batch_id)
        break
        # src = input_batch
        # test = src.permute(1, 0, 2)
        # plt.plot(test[:, 0, :].reshape(-1))
        # plt.plot(test[:, 1, :].reshape(-1))
        # plt.plot(test[:, 2, :].reshape(-1))

        # plt.savefig('test1.png')
