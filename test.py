import os
# the mode argument of the os.makedirs function may be ignored on some systems
# umask (user file-creation mode mask) specify the default denial value of variable mode,
# which means if this value is passed to makedirs function,
# it will be ignored and a folder/file with d_________ will be created
# we can either set the umask or specify mode in makedirs

# oldmask = os.umask(0o770)

# 避免使用对于时间戳之前的尾市体，使用原始的代码。 2-3是在2的基础上做出的还原。为了控制变量
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder, Measure, save_config, get_git_version_short_hash, get_git_description_last_commit, load_checkpoint, new_checkpoint
from model import xERTE
from segment import *
from database_op import DBDriver

def reset_time_cost():
    return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float),
            'data': defaultdict(float)}


def str_time_cost(tc):
    if tc is not None:
        data_tc = ', '.join('data.{} {:3f}'.format(k, v) for k, v in tc['data'].items())
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def prepare_inputs(contents, dataset='train', start_time=0, tc=None):
    '''
    :param tc: time recorder
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :param dataset: 'train', 'valid', 'test'
    :return:
    events concatenated with negative sampling
    '''
    t_start = time.time()
    if dataset == 'train':
        contents_dataset = contents.train_data
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'valid':
        contents_dataset = contents.valid_data
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'test':
        contents_dataset = contents.test_data
        assert start_time < max(contents_dataset[:, 3])
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset if event[3] >= start_time])
    if args.timer:
        tc['data']['load_data'] += time.time() - t_start
    return events


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0], dtype=np.int32)
        self.rel_idx = np.array(transposed_data[1], dtype=np.int32)
        self.target_idx = np.array(transposed_data[2], dtype=np.int32)
        self.ts = np.array(transposed_data[3], dtype=np.int32)
        self.event_idx = np.array(transposed_data[-1], dtype=np.int32)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.target_idx = self.target_idx.pin_memory()
        self.ts = self.ts.pin_memory()
        self.event_idx = self.event_idx.pin_memory()

        return self

    def __str__(self):
        return "Batch Information:\nsrc_idx: {}\nrel_idx: {}\ntarget_idx: {}\nts: {}".format(self.src_idx, self.rel_idx, self.target_idx, self.ts)


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim', type=int, default=[512, 10, 10, 10], nargs='+', help='dimension of embedding for node, realtion and time')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--DP_steps', type=int, default=1, help='number of DP steps')
parser.add_argument('--DP_num_edges', type=int, default=[200,15,15], nargs='+', help='number of edges at each sampling') # nargs='+' 表示长度不固定
parser.add_argument('--max_attended_edges', type=int, default=60, help='max number of edges after pruning')
parser.add_argument('--ratio_update', type=float, default=0.75, help='ratio_update: when update node representation: '
                                                                  'ratio * self representation + (1 - ratio) * neighbors, '
                                                                  'if ratio==0, GCN style, ratio==1, no node representation update')
parser.add_argument('--dataset', type=str, default='ICEWS14_forecasting', help='specify data set')
parser.add_argument('--whole_or_seen', type=str, default='whole', choices=['whole', 'seen', 'unseen'], help='test on the whole set or only on seen entities.')
parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--device', type=int, default=0, help='-1: cpu, >=0, cuda device')
parser.add_argument('--sampling', type=int, default=3,
                    help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors, 3: time-difference weighted')
parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
parser.add_argument('--weight_factor', type=float, default=2, help='sampling 3, scale the time unit')
parser.add_argument('--node_score_aggregation', type=str, default='sum', choices=['sum', 'mean', 'max'])
parser.add_argument('--ent_score_aggregation', type=str, default='sum', choices=['sum', 'mean'])
parser.add_argument('--emb_static_ratio', type=float, default=1, help='ratio of static embedding to time(temporal) embeddings')
parser.add_argument('--add_reverse', action='store_true', default=True, help='add reverse relation into data set')
parser.add_argument('--loss_fn', type=str, default='BCE', choices=['BCE', 'CE'])
parser.add_argument('--no_time_embedding', action='store_true', default=False, help='set to stop use time embedding')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--sqlite', action='store_true', default=None, help='save information to sqlite')
parser.add_argument('--mongo', action='store_true', default=None, help='save information to mongoDB')
parser.add_argument('--use_database', action = 'store_true', default=None, help='use database to store experimental')
parser.add_argument('--gradient_iters_per_update', type=int, default=1, help='gradient accumulation, update parameters every N iterations, default 1. set when GPU memo is small')
parser.add_argument('--timer', action='store_true', default=None, help='set to profile time consumption for some func')
parser.add_argument('--debug', action='store_true', default=None, help='in `debug` mode, checkpoint will not be saved')
parser.add_argument('--diac_embed', action='store_true',default=True, help='use entity-specific frequency and phase of time embeddings')
parser.add_argument('--reg_weight', type=float, default=1e-2, help='reg_weight')

parser.add_argument('--loss_error_weight', type=float, default=1, help='loss_error_weight')
parser.add_argument('--loss_margin', type=float, default=0.5, help='loss_margin')


args = parser.parse_args()

if not args.debug:
    import local_config
    save_dir = local_config.save_dir
else:
    save_dir = ''

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(args)

    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    # profile time consumption
    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()

    # init model and checkpoint folder
    start_time = time.time()
    struct_time = time.gmtime(start_time)
    epoch_command = args.epoch

    if args.load_checkpoint is None:
        checkpoint_dir, CHECKPOINT_PATH = new_checkpoint(save_dir, args, struct_time)
        contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

        adj = contents.get_adj_dict()  #主语：【谓词，谓语，时间】
        rel_adj = contents.get_rel_dict() #关系：尾市体，时间
        max_time = max(contents.data[:, 3])

        # construct NeighborFinder
        if 'yago' in args.dataset.lower():
            time_granularity = 1
        elif 'icews' in args.dataset.lower():
            time_granularity = 24
        else:
            raise ValueError
        nf = NeighborFinder(adj,rel_adj=rel_adj, sampling=args.sampling, max_time=max_time, num_entities=contents.num_entities,num_relations=contents.num_relations,
                            weight_factor=args.weight_factor, time_granularity=time_granularity, name = args.dataset) #生成每个实体直接相连的实体；并且生成每个时间戳的prior time便于后面处理
        # construct model
        model = xERTE(nf, contents.num_entities, contents.num_relations, contents.timestamps, contents.ent_time_set, args.emb_dim, DP_steps=args.DP_steps,
                       DP_num_edges=args.DP_num_edges, max_attended_edges=args.max_attended_edges,
                       node_score_aggregation=args.node_score_aggregation, ent_score_aggregation=args.ent_score_aggregation,
                       ratio_update=args.ratio_update, device=device, diac_embed=args.diac_embed, emb_static_ratio=args.emb_static_ratio,
                       use_time_embedding=not args.no_time_embedding, loss_margin=args.loss_margin)
        # move a model to GPU before constructing an optimizer, http://pytorch.org/docs/master/optim.html
        model.to(device)
        model.entity_raw_embed.cpu()
        model.relation_raw_embed.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 0
        if not args.debug:
            print("Save checkpoints under {}".format(CHECKPOINT_PATH))
            
        # 原先的学习率策略
        # schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=496,gamma=0.2,last_epoch=-1) #指定步长，每个iteration  2e-3 4e-4  8e-5  1.6e-6
        # schedular.step()
        
        # 新的学习率策略
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[496, 1488, 2976],gamma=0.2,last_epoch=-1) #设定调整时机
        # schedular.step()
        
        # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=100,verbose=True,min_lr=2e-5)
        # schedular.step(loss)
        
    else:
        checkpoint_dir = os.path.dirname(args.load_checkpoint)

    best_epoch = 2
    best_val = 0



    os.system("python ./eval.py --load_checkpoint {}/checkpoints_2021_11_3_3_6_34/checkpoint_{}.pt --whole_or_seen {} --device {} --mongo".format(args.dataset,
                                                    best_epoch, args.whole_or_seen, args.device))
