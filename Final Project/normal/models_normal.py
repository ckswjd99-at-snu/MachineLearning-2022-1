import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CustomCNN(nn.Module):

    def weight_init(self, seq):
        for m in seq.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, output_size):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model        
        self.output_size = output_size

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, output_size),
        )

        self.weight_init(self.l1)
        self.weight_init(self.l2)
        self.weight_init(self.l3)
        self.weight_init(self.l4)
        

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
    
    def forward(self, inputs):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-2: code CNN forward path

        # inputs: List[ Tensor(seq_len, 1, 28, 28) * batch_size ]
        batch_size = len(inputs)

        result = []

        for input in inputs:
            input = input.to("cuda:0")  # input: Tensor(seq_len, 1, 28, 28)
            x = self.l1(input)
            x = self.l2(x)
            x = self.l3(x)
            x = self.l4(x)
            x = x.view(x.size(0), -1)   # x: Tensor(seq_len, cnn_output_size)
            result.append(x)
        # result: List[ Tensor(seq_len, cnn_output_size) * batch_size ]
        
        outputs = result

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        # define the properties
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-1: Define lstm and input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class
        
        # you can either use torch LSTM or manually define it
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=0., 
            batch_first=True
        )
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h_curr=None, c_curr=None):
        """
        For reference (shape example)
        feature: List[ Tensor(seq_len, cnn_output_size) * batch_size ]
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-2: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various        
        if h_curr is None:
            output, (h_next, c_next) = self.lstm(feature)
        else:
            output, (h_next, c_next) = self.lstm(feature, (h_curr, c_curr))

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        
        return output, h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, cnn_output_size=256, rnn_num_layers=1):
        # NOTE: you can freely add hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_output_size = cnn_output_size
        self.rnn_input_size = cnn_output_size
        self.rnn_hidden_size = cnn_output_size
        self.rnn_num_layers = rnn_num_layers

        self.conv = CustomCNN(output_size=self.cnn_output_size)
        self.lstm = LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, 26)

        self.rnn_h_first_initializer = nn.Linear(self.cnn_output_size, self.rnn_hidden_size)
        self.rnn_h_second_initializer = nn.Linear(self.cnn_output_size, self.rnn_hidden_size)


    def forward(self, inputs):
        
        images = inputs
        # images: List[ Tensor(seq_len, 1, 28, 28) * batch_size ]

        batch_size = len(images)

        cnn_output = self.conv(images)
        # cnn_output: List[ Tensor(seq_len, cnn_output_size) * batch_size ]

        cnn_output_first = torch.cat([co[:1, :] for co in cnn_output]).reshape(batch_size, 1, self.cnn_output_size).to('cuda:0')
        # Tensor(batch_size, 1, cnn_output_size)
        cnn_output_second = torch.cat([co[1:2, :] for co in cnn_output]).reshape(batch_size, 1, self.cnn_output_size).to('cuda:0')
        # Tensor(batch_size, 1, cnn_output_size)

        '''
        # INITIALIZING LSTM WITH FRONT SEQENCE FEATURES
        rnn_h_first_init = self.rnn_h_first_initializer(cnn_output_first).transpose(0, 1).to('cuda:0')
        # Tensor(1, batch_size, rnn_hidden_size)

        rnn_h_second_init = self.rnn_h_second_initializer(cnn_output_second).transpose(0, 1).to('cuda:0')
        # Tensor(1, batch_size, rnn_hidden_size)

        rnn_h_init = torch.cat([rnn_h_first_init, rnn_h_second_init])
        # Tensor(2, batch_size, rnn_hidden_size)
        
        rnn_c_init = torch.zeros(rnn_h_init.shape).to('cuda:0')
        # Tensor(2, batch_size, rnn_hidden_size)
        '''

        padded_features = torch.nn.utils.rnn.pad_sequence(cnn_output, batch_first=True)
        # padded_features: Tensor(batch_size, max_seq_len, cnn_output_size)

        seq_lens = [i.shape[0] for i in images]
        # seq_lens: List[ <int> * batch_size ]

        packed_features = torch.nn.utils.rnn.pack_padded_sequence(padded_features, seq_lens, batch_first=True, enforce_sorted=False)
        # PackedSequence()

        packed_lstm_output, _, _ = self.lstm(packed_features)
        # packed_lstm_output, _, _ = self.lstm(packed_features, rnn_h_init, rnn_c_init)

        lstm_output, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_output, batch_first=True)
        # lstm_output: Tensor(batch_size, max_seq_len, rnn_output_size)
        # seq_lens: Tensor(batch_size)
        
        lstm_output = [lo[:seq_lens[idx]]for idx, lo in enumerate(lstm_output)]
        # List[ Tensor(seq_len, rnn_output_size) * batch_size ]

        classified_bag = []
        for lo in lstm_output:
            classified = self.classifier(lo)
            classified_bag.append(classified)
        # classified_bag: List[ Tensor(seq_len, 26) * batch_size ]

        outputs = classified_bag    # List[ Tensor(seq_len, 26) * batch_size ]
        return outputs


