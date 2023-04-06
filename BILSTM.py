from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import loadData
from EarlyStopping import EarlyStopping
import resource
import time
import datetime
torch.cuda.set_device(9)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dec_size, batch_size, latent_dim, n_layers=1, dropout_p=0.1, \
                 device = None):

        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.Nz = latent_dim
        self.hidden_dec_size = hidden_dec_size
        self.num_directions = 2

        self.device = device
        # self.rnn_dir = rnn_dir
        # self.bi_mode = bi_mode

        # self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, \
        #                    batch_first=True, bidirectional=rnn_dir == 2)
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=True)

        #self.initial = nn.Linear(self.Nz, hidden_dec_size * 2)
        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
        # self.mu = nn.Linear(hidden_size , self.Nz)
        # self.sigma = nn.Linear(hidden_size , self.Nz)

    def forward(self, inp_enc):
        # print(inp_enc.shape)
        output, (hidden, cell_state) = self.rnn(inp_enc)
        #output(seq_len, batch_size, num_directions * hidden_size)
        #h_n(num_directions * num_layers, batch_size, hidden_size)

        # hidden_cat = hidden.squeeze(0)
        #
        # initial_params = torch.tanh(self.initial[num_of_task](z))
        # (dec_hidden, dec_cell_state) = initial_params[:, :self.hidden_dec_size].contiguous(), initial_params[:,
        #                                                                                   self.hidden_dec_size:].contiguous()
        # # print('Encoder finished')
        return hidden, cell_state


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, latent_dim,  n_layers=1, dropout_p=0.1, device=None):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        # self.num_gaussian = num_gaussian
        self.Nz = latent_dim
        self.output_dim = output_size
        self.device = device
        self.num_directions = 2
        # self.cond_gen = cond_gen

        # if cond_gen:
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True,bidirectional=True)

        self.fc_out = nn.Linear(self.num_directions * self.hidden_size,  self.output_dim)
        # self.initial = nn.Linear(self.Nz, 2 * hidden_size)
        # else:
        # self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)

        # self.gmm = nn.Linear(hidden_size, num_gaussian * 6 + 3)

    def forward(self, inp_dec,hidden, cell):
        # input = [batch size, ]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context is z here = [n layers, batch size, hid dim]

        # if self.cond_gen:
        # print(z.shape)
        # print(inp_dec)
        # print(inp_dec.shape[1])

        # if self.training:
        # z_split = torch.stack([z.view(-1)] * (inp_dec.shape[1])).split(self.Nz, 1)
        # z_stack = torch.stack(z_split)
        # else:
        #     z_stack = z.unsqueeze(0)

        #inp_dec = inp_dec.unsqueeze(1)
        #inp_dec = torch.cat([inp_dec, z_stack], dim=2)
        inp_dec = inp_dec.unsqueeze(1)
        #print('inp_dec.shape',inp_dec.shape)#[32 1 1]

        output, (hidden, cell) = self.rnn(inp_dec, (hidden, cell))
        # print('output.shape',output.shape)#([32, 1, 200])
        # print('hidden.shape', hidden.shape)#[2 32 100]

        prediction = self.fc_out(output.squeeze(1))
        #print('prediction.shape', prediction.shape)#[32 1]
        # print(prediction)
        # print(prediction.shape)
        #
        # print('Decoder finished')
        return prediction, hidden, cell

class Bi_lstm(nn.Module):
    def __init__(self, encoder, decoder):
        super(Bi_lstm, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg):
        # src = [seq_len, batch size, input_size]##[60 32 1]
        # trg = [seq_len, batch size, input_size]

        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # hidden_enc = hidden_dec = self.encoder.initHidden()

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        # print(src)
        # print(np.shape(src))
        # print(batch_size)
        # print(max_len)

        trg_out = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_out).cuda()

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden, cell = self.encoder(src)
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        seed = 2022
        torch.manual_seed(seed)
        input = torch.rand(batch_size, self.encoder.input_size).cuda()#batch first=true
        # input = trg[0, :]
        # print(input)

        for t in range(0, max_len):
            # print("****")
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            #print('hidden.shape',hidden.shape)
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            # teacher_force = torch.rand(1,1) < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if t < max_len-1:
                #input = trg[t, :] if teacher_force else output
                input = output
        # print(outputs)
        return outputs


def MAELoss(output, target):
    return np.mean(np.mean(np.abs(target - output)))


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
            src = input_batch.to(device)#[60 32 1]
            trg = output_batch.to(device)#[1 32 1]

            # print(src.shape)
            # print(trg.shape)

            output = model(src, trg)  # turn off teacher forcing

            taskid = x_batch_id[0][0]
            batch_start = x_batch_id[0][1]
            batch_end = x_batch_id[-1][1]
            outputY[taskid, batch_start:batch_end + 1, :, :] = output.permute(1, 0, 2).detach().cpu().numpy()
            targetY[taskid, batch_start:batch_end + 1, :, :] = trg.permute(1, 0, 2).detach().cpu().numpy()

            # print(output.shape)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            # print(output)
            # print(trg)

            output = output.reshape(-1)
            trg = trg.reshape(-1)
            output = output.detach().cpu().numpy()
            trg = trg.detach().cpu().numpy()

            # print(trg.shape)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

    loss1 = MAELoss(outputY, targetY)
    loss2 = MAPELoss(outputY, targetY)
    loss3 = smape(targetY, outputY)
    loss4 = RMSE(outputY, targetY)

    # import matplotlib.pyplot as plt
    # for i in range(40):
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     ax.plot(outputY[i, :, 0, :].reshape(-1), 'k')
    #     ax.plot(targetY[i, :, 0, :].reshape(-1), 'g')
    #     plt.savefig('pics/dataset2/lstm_test{}-{}-{}.png'.format(i, testX.shape[2], testY.shape[2]))

    return loss1, loss2, loss3, loss4


if __name__ == '__main__':
    ### this procedure is for the future_step = 1
    num_epochs = 100
    #learning_rate = 3e-4
    learning_rate = 0.01
    #device = torch.device("cuda:9")
    # BATCH_SIZE = 128
    output_dir = '--output_dir'
    LATENT_CODE_SIZE = 32
    N_LAYERS=1
    INPUT_DIM=1
    OUTPUT_DIM=1
    NUM_circle = 1

    for set_num in [1,2]:
        r = []
        if set_num == 1:
            seq = [60, 90, 120]
            future = [1, 6, 12]
            NUM_TASKS = 100
            HID_DIM = 100
            HID_DEC_DIM = HID_DIM
            #teacher_forcing_ratio = 0.1
            BATCH_SIZE = 128

        elif set_num == 2:
            seq = [40, 60, 80]
            future = [1,4, 8]
            NUM_TASKS = 268
            HID_DIM = 100
            HID_DEC_DIM = HID_DIM
            #teacher_forcing_ratio = 0.8
            BATCH_SIZE = 32
        else:
            pass
        for seq_length in seq:
            for future_length in future:
                f = open('./loss/BILSTM_train_loss.txt', 'a')
                print('set_num', set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                f.close()
                f = open('./loss/BILSTM_valid_loss.txt', 'a')
                print('set_num', set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                f.close()
                res = [0, 0, 0, 0]
                for num in range(NUM_circle):
                    start = datetime.datetime.now()
                    trainX, trainY, validX, validY, testX, testY = loadData.load(1, seq_length, future_length, set_num)
                    print(validX.shape, validY.shape)
                    enc = Encoder(INPUT_DIM, HID_DIM, HID_DEC_DIM, BATCH_SIZE, LATENT_CODE_SIZE)
                    dec = Decoder(INPUT_DIM, HID_DEC_DIM, OUTPUT_DIM, BATCH_SIZE, LATENT_CODE_SIZE)
                    # print('before create model...')
                    # print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # 1018940 (kb)
                    lstm = Bi_lstm(enc, dec)
                    #lstm = Bi_lstm(num_classes, input_size, hidden_size, num_layers, dropout_p)
                    device = torch.device("cuda:9")
                    lstm.to(device)

                    criterion = torch.nn.MSELoss()  # mean-squared error for regression
                    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
                    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
                    train_time = 0
                    epoch_num = 0
                    es = EarlyStopping(patience=5)
                    begin_train = datetime.datetime.now()
                    # Train the model
                    for epoch in range(num_epochs):
                        start_time = time.time()
                        epoch_loss = 0
                        lstm.train()
                        batch_i = 0
                        for total_id, (input_batch, output_batch, epoch_completed, x_batch_id) in \
                                enumerate(loadData.get_batches(trainX, trainY, BATCH_SIZE)):
                            optimizer.zero_grad()
                            src = input_batch.to(device)
                            trg = output_batch.to(device)
                            batch_size = trg.shape[1]

                            outputs = lstm(src, trg)
                            # obtain the loss function
                            loss = criterion(outputs, trg)
                            epoch_loss += loss
                            batch_i += 1
                            loss.backward()
                            optimizer.step()
                        epoch_loss = epoch_loss / batch_i
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                        train_time += epoch_secs
                        validLoss = evaluate(lstm, validX, validY, device)
                        print('Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s'.format(epoch=epoch, epoch_mins= epoch_mins, epoch_secs=epoch_secs))
                        print('\tTrain Loss: {train_loss:.3f}'.format(train_loss=epoch_loss.item()))
                        print('\t Val. Loss: {valid_loss}'.format(valid_loss=validLoss))
                        f = open('./loss/BILSTM_train_loss.txt', 'a')
                        #print('seq_length ', seq_length, ' future_length', future_length, file=f)
                        print('num',num,'Epoch:', epoch, '\tTrain Loss: {train_loss:.3f}'.format(train_loss=epoch_loss.item()), file=f)
                        f.close()
                        f = open('./loss/BILSTM_valid_loss.txt', 'a')
                        #print('seq_length ', seq_length, ' future_length', future_length, file=f)
                        print('num',num,'Epoch:', epoch, 'valid_loss:', validLoss, file=f)
                        f.close()
                        if es.step(validLoss[0]):
                            print("Early Stopping")
                            break
                        torch.save(lstm.state_dict(),
                                   'saveModels/lstm/BILSTM-tut1-model-set_num{}-num{}--seq{}-future{}.pt'.format(set_num,num,seq_length,

                                                                                                  future_length))
                        epoch_num = epoch + 1
                    end_train = datetime.datetime.now()
                    train_time_each_epoch = (end_train - begin_train) / epoch_num
                    print('train time consuming：{}s'.format(train_time_each_epoch.total_seconds()))
                    #*******test***************
                    start_test = datetime.datetime.now()
                    testLoss = evaluate(lstm, testX, testY, device)
                    end_test = datetime.datetime.now()
                    test_time = end_test - start_test
                    # ****average test loss********
                    # tmp_list = list(testLoss)
                    # for j in range(len(tmp_list)):
                    #     res[j] += tmp_list[j] / NUM_circle
                    # ********print******************
                    print('\t Test. Loss: {test_loss}'.format(test_loss=testLoss))
                    print('test time consuming：{}s'.format(test_time.total_seconds()))
                    f = open('./loss/BILSTM_test_loss.txt', 'a')
                    print('set_num',set_num,'num',num,'seq_length ', seq_length, ' future_length', future_length, file=f)
                    print('test_loss:', testLoss, 'train time consuming：{}s'.format(train_time_each_epoch.total_seconds()),
                          'test time consuming：{}s'.format(test_time.total_seconds()),file=f)
                    f.close()
                    end = datetime.datetime.now()
                    total_time = end - start
                    # f = open('./loss/BILSTM_memory.txt', 'a')
                    # print('set_num', set_num, 'num', num, 'seq_length ', seq_length, ' future_length', future_length,
                    #       file=f)
                    # print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=f)
                    # print('total_time', total_time, file=f)
                    # f.close()
                    # # r.append(testLoss)
                    # # r.append(test_loss[2])
                    # np.savetxt('BILSTM_dataset2_result.csv', np.array(r).T, delimiter=',')
                # f = open('./loss/BILSTM_averg_test_loss.txt', 'a')
                # print('NUM_circle', NUM_circle, 'set_num',set_num, 'seq_length ', seq_length, ' future_length', future_length, file=f)
                # print('res', res, file=f)
                # f.close()