import torch.nn as nn
import torch
class LSTM(nn.Module):
    """
    This is the simple RNN model we will be using to perform Financial Time-series Analysis.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        # Initialize hidden state with zeros
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :]) 
        
        return out