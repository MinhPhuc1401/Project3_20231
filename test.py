import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from exp.exp_model import Exp_Model
from data_load.data_loader import Dataset_Custom

if __name__ == '__main__':

    fix_seed = 100
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='generating')

    # Load data
    parser.add_argument('--root_path', type=str, default='./data/2016', help='root path of the data files')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--sequence_length', type=int, default=20, help='length of input sequence')
    parser.add_argument('--prediction_length', type=int, default=None, help='prediction sequence length')
    parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
    parser.add_argument('--input_dim', type=int, default=6, help='dimension of input')
    parser.add_argument('--hidden_size', type=int, default=128, help='encoder dimension')
    parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

    # Diffusion process
    parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
    parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
    parser.add_argument('--beta_end', type=float, default=1.0, help='end of the beta')
    parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

    # Bidirectional VAE
    parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')
    parser.add_argument('--mult', type=float, default=1, help='mult of channels')
    parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
    parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
    parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
    parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
    parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--itr', type=int, default=1, help='experiment times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
    parser.add_argument('--zeta', type=float, default=0.5, help='trade off parameter zeta')
    parser.add_argument('--eta', type=float, default=1.0, help='trade off parameter eta')

    # Device
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.prediction_length is None:
        args.prediction_length = args.sequence_length

    print('Args in experiment:')
    print(args)


    model = Exp_Model(args)


    def load_and_predict(selected_ticker):
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size

        data_set = Dataset_Custom(
            root_path=args.root_path,
            data_path=selected_ticker + ".csv",
            flag="test",
            size=[args.sequence_length, args.prediction_length],
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        noisy = []
        preds = []
        trues = []
        input = [] 
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if i==2:
                break
            batch_x = batch_x.float().to(model.device)
            batch_y = batch_y[...,-model.args.target_dim:].float().to(model.device)
            batch_x_mark = batch_x_mark.float().to(model.device)
            noisy_out, out = model.pred_net(batch_x, batch_x_mark)
            noisy.append(noisy_out.squeeze(1).detach().cpu().numpy())
            preds.append(out.squeeze(1).detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            input.append(batch_x[...,-1:].detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        noisy = np.array(noisy)

        # preds: (batch, seq_len, 1)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues: (batch, seq_len, 1)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # noisy: (batch, seq_len, 1)
        noisy = noisy.reshape(-1, noisy.shape[-2], noisy.shape[-1])
        # preds = preds + noisy
        return preds, trues

    preds, trues = load_and_predict("AAPL")
    def visualize_predictions(preds, trues):
        # Assuming preds and trues are numpy arrays with shape (num_samples, sequence_length, 1)

        for i in range(len(preds)):
            plt.figure(figsize=(10, 6))

            # Plotting true values
            plt.plot(trues[i], label='True', marker='o')

            # Plotting predicted values
            plt.plot(preds[i], label='Predicted', marker='o')

            plt.title(f'Stock Price Prediction - Sample {i + 1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.grid(True)
            plt.show()

    visualize_predictions(preds, trues)