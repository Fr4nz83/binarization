import argparse

from binary_modules import binarylinear

def get_parser():
    parser = argparse.ArgumentParser(description='MLP-Mixer Compression for Image Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', '-d',
                        default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="Dataset name")

    parser.add_argument('--image-dir',
                        type=str,
                        default="./imagenet",
                        help="Path to imagenet dir (not required for CIFAR")

    parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16",
                                                 "Mixer-B_16-21k", "Mixer-L_16-21k"],
                        default="Mixer-B_16",
                        help="Which model to use.")
    parser.add_argument("--pretrained-dir", type=str, default="checkpoint/Mixer-B_16.npz",
                        help="Path to the pre-trained model.")

    parser.add_argument('-j', '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs',
                        default=400,
                        type=int,
                        help='number of total epochs to run')

    parser.add_argument("--train-batch-size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval-batch-size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--lr', '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')

    parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')

    parser.add_argument('--weight-decay', '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay')

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=None, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--amp', default=False, action="store_true")

    parser.add_argument('--binary', default=False, action="store_true")

    parser.add_argument('--binary-mode', choices=binarylinear.__all__)
    parser.add_argument('--binary-activation', choices=["GELU", "Hardtanh"])

    parser.add_argument(
        '--tau_min',
        default=0.85,
        type=float,
        help='tau_min')

    parser.add_argument(
        '--tau_max',
        default=0.99,
        type=float,
        help='tau_max')


    parser.add_argument('--prune', default=False, action="store_true")

    parser.add_argument('--percentage', type=float, default=0.9)

    parser.add_argument('--resume', type=str)

    parser.add_argument('--optim', type=str, choices=["adam", "sgd"], default="sgd")

    parser.add_argument('--name',
                        type=str,
                        help="The name of the directory where the model will be saved")

    parser.add_argument('--out-dir', '-o',
                        dest='output_dir',
                        default='logs',
                        help='Path to dump logs and checkpoints')



    return parser
