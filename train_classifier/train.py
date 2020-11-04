import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTM

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(model_info['input_dim'], model_info['hidden_dim'], model_info['num_layers'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    train_y = train_data[[0]].values
    train_y = torch.from_numpy(train_y).float()
    
    train_X = train_data.drop([0], axis=1).values
    train_X = torch.from_numpy(train_X).float().unsqueeze(1)
    print(train_X.size())
    print(train_y.size())

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        hidden = None
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad()
            prediction = model(batch_X)
            loss = loss_fn(prediction, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
        if epoch % 50 == 0:
            print("Epoch: {}, MSELoss: {};".format(epoch, total_loss / len(train_loader)))
    return model

def validate(trained_model, val_data_dir, loss_fn):
    data = pd.read_csv(os.path.join(val_data_dir, "test.csv"), header=None, names=None)
    
    y = data[[0]].values
    y = torch.from_numpy(y).float()
    
    X = data.drop([0], axis=1).values
    X = torch.from_numpy(X).float().unsqueeze(1)

    loss = loss_fn(trained_model.forward(X), y)
    
    return loss

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--input_dim', type=int, default=32, metavar='N',
                        help='size of the input (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of layers (default: 1)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='N',
                        help='number of outputs (default: 1)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--val-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTM(args.input_dim, args.hidden_dim,args.num_layers, args.output_dim).to(device)

    print("Model loaded with input_dim {}, hidden_dim {}, num_layers{},output_dim {}.".format(
        args.input_dim, args.hidden_dim, args.num_layers,args.output_dim
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss(size_average=True)

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    print("val_MSE: {};".format(validate(model, args.val_data_dir, loss_fn)))
    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
        
    # Save the word_dict

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)