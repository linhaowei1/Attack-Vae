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
    parser.add_argument('--mode', type=str, default='train',
                        help='choose to train or eval')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset')
    parser.add_argument('--batch_size', type=int, default=144,
                        help='batch size for training')
    parser.add_argument('--save_path', type=str, default='checkpoint',
                        help='save the model here')
    parser.add_argument('--log_file', type=str, default='log',
                        help='save logs')
    parser.add_argument('--params', type=str, default='/home/linhw/myproject/Attack-Vae/checkpoint/model/500epoch_model.pt',
                        help='load pretrained model')
    parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--model', default='smallCNN',type=str, required=False, help='Model Type')
    parser.add_argument('--seen', default='012345',type=str, required=False, help='seen classes')

    return parser.parse_args()