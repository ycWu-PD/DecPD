import argparse
import numpy as np
import torch


def set_random_seed(seed):
    """Set random seeds."""
    np.random.seed(seed)  # NumPy 随机库的种子
    torch.manual_seed(seed)  # PyTorch 随机库的种子
    torch.cuda.manual_seed(seed)  # CUDA 设备设置种子
    torch.cuda.manual_seed_all(seed)  # CUDA 设备设置种子


def args_parser():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument('--random_Seed', type=int, default=41, help="random seed")
    parser.add_argument('--sequence_length', type=int, default=326, help='Time slice length')
    parser.add_argument('--input_size', type=int, default=36, help="Input dimension/characteristic dimension")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden layer")
    parser.add_argument('--num_layers', type=int, default=5, help="LSTM layers")
    parser.add_argument('--num_classes', type=int, default=7, help="the amount of classification")
    parser.add_argument('--par', type=int, default=4, help="Peak factor value")
    parser.add_argument('--a', type=int, default=0, help="Focal Loss hyperparameter")

    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_epochs', type=int, default=300, help="Round by round")
    parser.add_argument('--learning_rate', type=int, default=0.01, help="Learning rate")

    args = parser.parse_args()

    return args