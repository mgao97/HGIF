


def parser_add_main_args(parser):
    # dataset and protocol
    parser.add_argument('--data_dir', type=str, default='../data') # need to be specified
    # parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--dataset', type=str, default='twibot-20')
    parser.add_argument('--save_dir', type=str, default='./pytorch_models/') # need to be specified
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    # model

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--C', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--sparse', type=int, default=1)

    parser.add_argument('--normalization', type=str, default='sym')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--test_epochs', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--display_step', type=int,
                        default=5, help='how often to print')
    parser.add_argument('--batch_size', type=int, default=1024)


    parser.add_argument('--temperature', type=float,default=0.5, help='how often to print')


    # for graph edit model
    parser.add_argument('--K', type=int, default=5,
                        help='num of views for data augmentation')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight for mean of risks from multiple domains')

