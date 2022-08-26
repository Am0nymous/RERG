import os
import sys
sys.path.append('./')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import gc
from utils.bert_torch import BertModel
from transformers import BertConfig
import time
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_constant_schedule, \
    get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, f1_score
from utils.models import LSTMWrapper, FC_net, MultiHeadAttention_graph
from utils.data_function import prepare_data
from torch.nn import CrossEntropyLoss
import unicodedata
import torch.nn.functional as F
from argparse import ArgumentParser
import torch
import random
import torch.nn as nn
import torch.distributed
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm, trange
import os
logger = logging.getLogger(__name__)
import numpy as np
from torch.distributions import Categorical

def parse_train_arg():
    parser = ArgumentParser()
    parser.add_argument('--hidden_size', default=1024, type=int, help="the dimension in the model")
    parser.add_argument('--layers', default=3, type=int, help="the number of attention layer in the attention module")
    parser.add_argument('--class_num', default=2, type=int, help="the number of classes in the relevant task")
    parser.add_argument('--do_train', default=True, help='Wether to run training')
    parser.add_argument('--do_eval', default=True, help='Wether to run eval on the dev dataset')
    parser.add_argument('--do_validation', default=False, help='wether to validate the saved model')
    parser.add_argument('--data_dir', default="./TABFACT_tapas_data/",
                        type=str, help='The input data dir')
    parser.add_argument('--output_dir', default="./output/large/", type=str, help='The output dir')
    parser.add_argument('--test_set', default='dev', type=str, choices=["dev", "test", "simple_test", "complex_test"],
                        help='Which test set is used for evaluation')
    parser.add_argument('--fp16', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Total batch size for eval')
    parser.add_argument('--max_batch_size', type=int, default=4, help="the batch_size in the task")
    parser.add_argument('--seed', type=int, default=9527, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--cpu", default=False, help="Whether not to use CUDA when available")
    parser.add_argument('--train_batch_size', default=4, type=int, help='Total batch size for training')
    parser.add_argument('--num_train_epochs', default=5, type=float, help='Total number of training epochs')
    parser.add_argument('--no_cuda', default=False, help='Whether not to use CUDA when available')
    parser.add_argument('--local_rank', default=[0], help='local rank for multi gpus')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', '--learning-rate', default='1e-5', type=float,
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--log_dir', default="./Running_info/", type=str)
    parser.add_argument("--model_path", default="./convert_model/tapas_model/", type=str)
    parser.add_argument('--device', default=torch.device("cuda:0"))
    parser.add_argument('--loss_type', type=str, default='CrossEntropy')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--scheduler', type=str, default='warmupcosine')

    args = parser.parse_args()
    return args


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)

    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }



class FaBERT(nn.Module):
    def __init__(self, config):
        super(FaBERT, self).__init__()
        self.tbert = BertModel.from_pretrained(config["model_path"])
        self.fc = nn.Linear(config["hidden_size"], config["class_num"])
        self.encoders = nn.ModuleList([MultiHeadAttention_graph(16, 1024, 64, 64, dropout=0.1) for _ in
                                       range(6)])

        self.edge_weight_tensor_k = torch.Tensor(100, 64)
        self.edge_weight_tensor_k = nn.Parameter(nn.init.kaiming_uniform_(self.edge_weight_tensor_k))
        self.edge_weight_tensor_v = torch.Tensor(100, 64)
        self.edge_weight_tensor_v = nn.Parameter(nn.init.kaiming_uniform_(self.edge_weight_tensor_v))

        self.sent_lstm = LSTMWrapper(input_dim=1024,
                                     hidden_dim=1024,
                                     n_layer=1,
                                     dropout=0.1)
        self.fc_lstm = nn.Linear(2 * config["hidden_size"], config["hidden_size"])

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, adj_matrix, graph_mask, bert_words,
                bert_maps, word_masks, start_pos, end_pos, flag, fc_net, label_id=0):

        vec_adj_k = F.embedding(adj_matrix.long(), self.edge_weight_tensor_k, padding_idx=0)
        vec_adj_v = F.embedding(adj_matrix.long(), self.edge_weight_tensor_v, padding_idx=0)

        word_output, _ = self.tbert(input_ids, attention_mask, token_type_ids, position_ids, return_dict=False)
        max_len = torch.sum(attention_mask, 1).max()
        batch, bert_dim = word_output.size(0), word_output.size(-1)
        out_features = torch.Tensor(batch, max_len, bert_dim).fill_(0)
        device = word_output.get_device() if word_output.is_cuda else None
        if device is not None:
            out_features = out_features.to(device)

        len_list = []
        for i in range(input_ids.size(0)):
            index = torch.arange(0, input_ids.size(-1)).cuda()
            index_ = index * attention_mask[i]
            non_zeros = index_.nonzero()
            out_features[i, :non_zeros.size(0), :] = word_output[i][non_zeros.squeeze(-1)]
            len_list.append(non_zeros.size(0))

        sorted_seq_lengths, indices = torch.sort(torch.tensor(len_list), descending=True)
        out_features = out_features[indices]
        out_features = self.sent_lstm(out_features, sorted_seq_lengths)
        _, desorted_indices = torch.sort(indices, descending=False)
        out_features = out_features[desorted_indices]
        ## get node representation
        max_node_num = graph_mask.size(1)
        graph = torch.Tensor(batch, max_node_num, bert_dim).fill_(0)
        device = word_output.get_device() if word_output.is_cuda else None
        if device is not None:
            graph = graph.to(device)
        for i in range(input_ids.size(0)):
            non_zeros_s = torch.cat((torch.tensor([0]).cuda(), start_pos[i].nonzero().squeeze(-1)))
            non_zeros_e = torch.cat((torch.tensor([0]).cuda(), end_pos[i].nonzero().squeeze(-1)))
            start_indices = start_pos[i][non_zeros_s]
            end_indices = end_pos[i][non_zeros_e]
            start_info = out_features[i][start_indices.long()][:, 1024:]
            end_info = out_features[i][end_indices.long()][:, :1024]
            graph[i][:len(non_zeros_s)] = self.fc_lstm(torch.cat((start_info, end_info), 1))

 
        action_probs = []
        entropys = []
        rp_loss = []
        for i, layer in enumerate(self.encoders):
            probs = getattr(fc_net, "fc{}".format(i))(graph)
            probs = torch.softmax(probs, dim=-1)
            m = Categorical(probs)
            if flag == 'three':
                action = probs.argmax(dim=-1)
            else:
                action = m.sample()

            ratio = (torch.sum(action, dim=-1).float() / adj_matrix.size(-1)).mean()
            rp_loss.append(1 + torch.max(torch.tensor(ratio - 0.60).cuda(), torch.tensor(0.).cuda()))
            entropys.append(m.entropy().sum())
            adj_matrix[:, 0, 0, :] = action
            adj_matrix[:, 1, 0, :] = action
            type_1, _ = layer(graph, graph, graph, vec_adj_k[:, 0, :, :], vec_adj_v[:, 0, :, :],
                              adj_matrix[:, 0, :, :],
                              adj_matrix[:, 0, :, :],
                              graph_mask.unsqueeze(1))
            type_2, _ = layer(graph, graph, graph, vec_adj_k[:, 1, :, :], vec_adj_v[:, 1, :, :],
                              adj_matrix[:, 1, :, :],
                              adj_matrix[:, 1, :, :],
                              graph_mask.unsqueeze(1))
            action_prob = m.log_prob(action)
            action_probs.append(action_prob)
            graph = (type_1 + type_2) / 2.0
        del type_1, type_2
        a = graph[:, 0].contiguous()
        logits = self.fc(a)
        return logits, action_probs, entropys, rp_loss


def evaluate(args, model, device, data_type, fc_net):
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    eval_dataloader, num_train_examples = prepare_data(args, data_type)
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    ground = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        flag = 'three'
        input_ids, attention_mask, segment_ids, column_ids, row_ids, prev_label_ids, column_ranks, inv_column_ranks, \
        numeric_relations, label_id, position_ids, adj_matrix, graph_mask, bert_words, bert_maps, word_masks, start_pos, end_pos = batch

        token_type_ids = (segment_ids, column_ids, row_ids, prev_label_ids, column_ranks, \
                          inv_column_ranks, numeric_relations)
        with torch.no_grad():
            logit, _, _, _ = model.forward(input_ids, attention_mask, token_type_ids,
                                           position_ids, adj_matrix, graph_mask.float(), bert_words, bert_maps,
                                           word_masks, start_pos, end_pos, flag, fc_net)

            loss = nn.CrossEntropyLoss()
            tmp_eval_loss = loss(logit, label_id.squeeze(1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds.extend(np.argmax(logit.detach().cpu().numpy(), axis=-1).tolist())
        labels = label_id.squeeze(1).detach().cpu().numpy().tolist()
        ground.extend(labels)
    eval_result = acc_and_f1(np.array(preds), np.array(ground))
    logging.info("{} result acc: {:.5f}".format(data_type, eval_result["acc"]))
    logging.info("{} result f1: {:.5f}".format(data_type, eval_result["f1"]))
    return eval_result


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                               num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                               num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                  num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))




def main():
    args = parse_train_arg()
    config = {}
    for item in args._get_kwargs():
        config[str(item[0])] = item[1]

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(args.log_dir)
    log_name = log_path + '/' + rq + '.log'
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("begining ...")

    # sys.stdout.flush()

    # set the gpu or multi gpus environment
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(init_method='tcp://localhost:26392', rank=0, world_size=1, backend='nccl')
        n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    logging.info("The seed is {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.barrier()

    logger.info("Datasets are loaded from {}\n Outputs will be saved to {}".format(args.data_dir, args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

    model = FaBERT(config)
    model.load_state_dict(torch.load(args.model_path)['net'])
    agent = FC_net()
    agent.load_state_dict(torch.load(args.model_path)['agent'])
    global_step = -1
    best_acc = -1
    if args.do_train:
        model.to(device)
        agent.to(device)

        model.eval()
        torch.set_grad_enabled(False)  # turn off gradient tracking
        logging.info("Evaluating the dev data .....")
        dev_result = evaluate(args, model, device, "dev", agent)
        writer.add_scalar('dev/acc', dev_result['acc'], global_step)

        output_dir = os.path.join(args.output_dir, 'save_step_{}_acc_{}'.format(global_step, best_acc))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.info("Testing the test data ....")
        test_result = evaluate(args, model, device, "test", agent)

        logging.info("Testing the simple_test data ....")
        simple_test_result = evaluate(args, model, device, "simple_test", agent)

        logging.info("Testing the small_test data ....")
        small_test_result = evaluate(args, model, device, "small_test", agent)

        logging.info("Testing the complex_test data ...")
        complex_test_result = evaluate(args, model, device, "complex_test", agent)

        evaluation_result = "dev_acc: {}, dev_f1: {}\n test_acc: {}, test_f1: {}\n \
                                         simple_test_acc: {}, simple_test_f1: {}\n small_test_acc: {}, small_test_f1: {}\n \
                                         complex_test_acc: {}, complex_test_f1: {}".format(
                    dev_result['acc'], dev_result["f1"], \
                    test_result["acc"], test_result["f1"], simple_test_result["acc"],
                    simple_test_result["f1"], \
                    small_test_result["acc"], small_test_result["f1"], complex_test_result["acc"],
                    complex_test_result["f1"])

        with open(os.path.join(output_dir, 'evaluation.txt'), 'w+') as f:
            f.write(evaluation_result)






if __name__ == "__main__":
    main()




