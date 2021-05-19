import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='cuda used for training')
    parser.add_argument('--batch-size', type=int, default=144,
                        help='batch size for training')
    parser.add_argument('--save_path', type=str, default='checkpoint',
                        help='save the model here')
    parser.add_argument('--log_file', type=str, default='log',
                        help='save logs')
    return parser.parse_args()