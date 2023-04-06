import numpy as np
import matplotlib

import loadData
import resource
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from EarlyStopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import time
import datetime


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class FNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size_1, hidden_size_2, dropout_p):
        super(FNN, self).__init__()

        self.regressor = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(input_size, hidden_size_1),#1-10 #[60 32 1]-[60 32 10]
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size_1, hidden_size_2),#10-10 [60 32 10]
            nn.Linear(hidden_size_2, num_classes)#10-1 [60 32 10]-[60 32 1]
        )

    def forward(self, x, y):
        max_len = y.shape[0]#y[6 32 1]--1
        output = self.regressor(x)#x[60 32 1]
        # print('in forword,output.shape,',output.shape)#[60 32 1]
        # print('in forword,output[-max_len:].shape,', output[-max_len:].shape)#[1 32 1]
        return output[-max_len:]#[-6:]


# model = FNN(1, 1, 20, 10, 0.5)
# input = torch.randn(10, 2, 1)
# target = torch.randn(5, 2, 1)
# output = model(input, target)
# print(output.shape)


def MAELoss(output, target):
    return np.mean(np.abs(output - target))


def MAPELoss(output, target):
    flag = target == 0
    return np.mean(np.abs((target - output) * (1 - flag) / (target + flag))) * 100


def smape(target, forcecast):
    denominator = np.abs(target) + np.abs(forcecast)
    flag = denominator == 0
    # print(1 - flag)
    smape = 2 * np.mean(
        (np.abs(target - forcecast) * (1 - flag)) / (denominator + flag)
    )
    return smape * 100


def RMSE(output, target):
    return np.sqrt(np.mean(np.square(target - output)))


def evaluate(model, testX, testY, device):
    model.eval()

    NUM_BATCH = testY.shape[1] // BATCH_SIZE
    outputY = np.ones_like(testY)[:, :NUM_BATCH * BATCH_SIZE, :, :]
    targetY = np.ones_like(testY)[:, :NUM_BATCH * BATCH_SIZE, :, :]

    with torch.no_grad():
        for batch_i, (input_batch, output_batch, epoch_completed, x_batch_id) in \
                enumerate(loadData.get_batches(testX, testY, BATCH_SIZE)):
            # print('batch_i',batch_i)
            # print('x_batch_id',x_batch_id)
            src = input_batch.to(device)#[60 32 1]
            trg = output_batch.to(device)#[1 32 1]

            # print(src.shape)#[60 32 1]
            # print(trg.shape)#[1 32 1]

            output = model(src, trg)  # turn off teacher forcing
            # print(output.shape)#

            taskid = x_batch_id[0][0]
            batch_start = x_batch_id[0][1]
            batch_end = x_batch_id[-1][1]
            outputY[taskid, batch_start:batch_end + 1, :, :] = output.permute(1, 0, 2).detach().cpu().numpy()
            targetY[taskid, batch_start:batch_end + 1, :, :] = trg.permute(1, 0, 2).detach().cpu().numpy()
            # print('taskid',taskid)
            # print(batch_start)
            # print(batch_end)
            # print('outputY.shape',outputY.shape)
            # print('targetY.shape',targetY.shape)
            # print('outputY', outputY)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            # print(output)
            # print(trg)

            # output = output.reshape(-1)#变成1行
            # trg = trg.reshape(-1)
            # output = output.detach().cpu().numpy()
            # trg = trg.detach().cpu().numpy()

            # print(trg.shape)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]


            del input_batch, output_batch, epoch_completed, x_batch_id, output
            torch.cuda.empty_cache()
    loss1 = MAELoss(outputY, targetY)
    loss2 = MAPELoss(outputY, targetY)
    loss3 = smape(targetY, outputY)
    loss4 = RMSE(outputY, targetY)

    # for i in range(40):
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     ax.plot(outputY[i, :, 0, :].reshape(-1), 'k')
    #     ax.plot(targetY[i, :, 0, :].reshape(-1), 'g')
    #     plt.savefig('pics/dataset2/fnn_test{}-{}-{}.png'.format(i, testX.shape[2], testY.shape[2]))

    return loss1, loss2, loss3, loss4


if __name__ == '__main__':
    num_epochs = 100
    # learning_rate = 3e-4
    learning_rate = 0.01

    input_size = 1
    hidden_size_1 = 10
    hidden_size_2 = 10
    num_classes = 1
    dropout_p = .1
    #device = torch.device("cuda:8")
    NUM_circle=1

    for set_num in [2]:
        r = []
        if set_num == 1:
            seq = [60, 90, 120]
            future = [1, 6, 12]
            BATCH_SIZE = 128
        elif set_num == 2:
            seq = [40, 60, 80]
            future = [1, 4, 8]
            BATCH_SIZE = 32#不可128
        else:
            pass
        for seq_length in seq:
            for future_length in future:
                f = open('./loss/fnn_train_loss.txt', 'a')
                print('set_num', set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                f.close()
                f = open('./loss/fnn_valid_loss.txt', 'a')
                print('set_num', set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                f.close()
                res = [0, 0, 0, 0]
                for num in range(NUM_circle):
                    start = datetime.datetime.now()
                    print('seq_length ', seq_length, ' future_length', future_length)
                    trainX, trainY, validX, validY, testX, testY = loadData.load(1, seq_length, future_length, set_num)

                    device = torch.device("cuda:8")

                    model = FNN(num_classes, input_size, hidden_size_1, hidden_size_2, dropout_p)#1 1 10 10 0.1
                    model.to(device)

                    criterion = torch.nn.MSELoss()  # mean-squared error for regression
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
                    #best_valid_loss = [1000, 1000, 1000]
                    # for i in range(3):
                    #     best_valid_loss[i]=1000
                    train_time = 0
                    epoch_num = 0
                    # Train the model
                    es = EarlyStopping(patience=10)
                    begin_train = datetime.datetime.now()
                    for epoch in range(num_epochs):
                        start_time = time.time()
                        print('epoch:', epoch)
                        epoch_loss = 0
                        model.train()
                        batch_i = 0
                        for total_id, (input_batch, output_batch, epoch_completed, x_batch_id) in \
                                enumerate(loadData.get_batches(trainX, trainY, BATCH_SIZE)):
                            #print('in data1 fnn,input_batch.shape',input_batch.shape)#([60, 32, 1])

                            # print(total_id)
                            # trg = [trg sent len, batch size]
                            # output = [trg sent len, batch size, output dim]#
                            optimizer.zero_grad()
                            src = input_batch.to(device)
                            trg = output_batch.to(device)
                            batch_size = trg.shape[1]
                            # print('src.shape',src.shape)#[60 32 1]
                            # print(trg.shape)#[1 32 1]

                            outputs = model(src, trg)
                            #print(outputs.shape)#[1 32 1]

                            # obtain the loss function
                            loss = criterion(outputs, trg)
                            epoch_loss += loss
                            batch_i += 1

                            loss.backward()

                            optimizer.step()

                            del input_batch, output_batch, epoch_completed, x_batch_id, outputs, loss
                            torch.cuda.empty_cache()
                        epoch_loss = epoch_loss / batch_i
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                        train_time += epoch_secs
                        validLoss = evaluate(model, validX, validY, device)
                        print(
                            'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s'.format(epoch=epoch, epoch_mins=epoch_mins,
                                                                                        epoch_secs=epoch_secs))
                        print('\tTrain Loss: {train_loss:.3f}'.format(train_loss=epoch_loss.item()))
                        print('\t Val. Loss: {valid_loss}'.format(valid_loss=validLoss))
                        f = open('./loss/fnn_train_loss.txt', 'a')
                        #print('set_num',set_num,'seq_length ', seq_length, ' future_length', future_length, file=f)
                        print('num',num,'Epoch:', epoch, 'Train Loss:', epoch_loss, file=f)
                        f.close()
                        f = open('./loss/fnn_valid_loss.txt', 'a')
                        #print('seq_length ', seq_length, ' future_length', future_length, file=f)
                        print('num',num,'Epoch:', epoch, 'valid_loss:', validLoss, file=f)
                        f.close()

                        if es.step(validLoss[0]):
                            print("Early stopping!")
                            break
                        torch.save(model.state_dict(),
                                   'saveModels/fnn/fnn-tut1-model-set_num{}-num{}--seq{}-future{}.pt'.format(set_num,num,seq_length,
                                                                                                  future_length))

                        epoch_num = epoch + 1
                        print('epoch_...', epoch_num)
                        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    # train_time_each_epoch = train_time/epoch_num
                    # print('train_time_each_epoch',train_time_each_epoch)

                    end_train = datetime.datetime.now()
                    train_time_each_epoch = (end_train - begin_train) / epoch_num
                    print('train time consuming：{}s'.format(train_time_each_epoch.total_seconds()))
                    # ************test*******************
                    start_test = datetime.datetime.now()
                    testLoss = evaluate(model, testX, testY, device)
                    end_test = datetime.datetime.now()
                    test_time = end_test - start_test
                    # ****average test loss********
                    tmp_list = list(testLoss)
                    for j in range(len(tmp_list)):
                        res[j] += tmp_list[j] / NUM_circle
                    #********print******************
                    print('test time consuming：{}s'.format(test_time.total_seconds()))
                    print('\t Test. Loss: {test_loss}'.format(test_loss=testLoss))
                    f = open('./loss/fnn_test_loss.txt', 'a')
                    print('set_num',set_num,'num',num,'seq_length ', seq_length, ' future_length', future_length, file=f)
                    print('test_loss:', testLoss, 'train time consuming：{}s'.format(train_time_each_epoch.total_seconds()),
                          'test time consuming：{}s'.format(test_time.total_seconds()), file=f)
                    f.close()
                    print('after test...')
                    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    end = datetime.datetime.now()
                    total_time = end - start
                    f = open('./loss/fnn_memory.txt', 'a')
                    print('set_num',set_num,'num',num,'seq_length ', seq_length, ' future_length', future_length, file=f)
                    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=f)
                    print('total_time', total_time, file=f)
                    f.close()
                # f = open('./loss/fnn_averg_test_loss.txt', 'a')
                # print('NUM_circle', NUM_circle, 'set_num',set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                # print('res', res, file=f)
                # f.close()


                    # r.append(testLoss)
                    # # r.append(test_loss[2])
                    # np.savetxt('fnn_dataset2_result.csv', np.array(r).T, delimiter=',')
    # data_predict = train_predict.data.numpy()

    # dataY_plot = dataY.data.numpy()
    #
    # data_predict = sc.inverse_transform(data_predict)
    # dataY_plot = sc.inverse_transform(dataY_plot)
    #
    # plt.axvline(x=train_size, c='r', linestyle='--')
    #
    # plt.plot(dataY_plot)
    # plt.plot(data_predict)
    # plt.suptitle('Time-Series Prediction')
    # plt.show()