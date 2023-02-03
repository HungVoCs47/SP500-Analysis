import torch
import torch.nn as nn




class LSTM(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 32, num_layers = 3, output_dim = 1, dropout = 0.1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.dropout_1 = nn.Dropout(dropout)

  

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm_1(x, (h0.detach(), c0.detach()))
        #h1 = torch.zeros(self.num_layers, out.size(0), self.hidden_dim).requires_grad_()
        #c1 = torch.zeros(self.num_layers, out.size(0), self.hidden_dim).requires_grad_()
        #out, (hn, cn) = self.lstm_2(out, (h1.detach(), c1.detach()))
        out = self.fc(out[:, -1, :]) 
        return out