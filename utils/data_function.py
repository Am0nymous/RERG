import os
import pickle
import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, RandomSampler, Dataset



class MyCollator(object):

    def __init__(self, types):
        self.type = types

    def __call__(self, data_mb):

        batch_size = len(data_mb)

        def get_batch_stat(data_mb):
            max_sent_num = 0
            max_input_num = 0
            for d in data_mb:
                if len(set(d["bert_maps"])) > max_sent_num:
                    max_sent_num = len(set(d["bert_maps"]))
            return max_sent_num

        max_nodes = get_batch_stat(data_mb)

        adj_matrix = np.zeros([batch_size, self.type, max_nodes, max_nodes])
        graph_mask = np.zeros([batch_size, max_nodes])

        input_ids = []
        input_mask = []
        segment_ids = []
        column_ids = []
        row_ids = []
        prev_label_ids = []
        column_ranks = []
        inv_column_ranks = []
        numeric_relations = []
        label_id = []
        position_id = []
        bert_words = []
        bert_maps = []
        start_poss = []
        end_poss = []
        word_masks = np.zeros([batch_size, len(data_mb[0]["input_ids"])])
        for index, per_data in enumerate(data_mb):
            input_ids.append(per_data["input_ids"])
            input_mask.append(per_data["input_mask"])
            segment_ids.append(per_data["segment_ids"])
            column_ids.append(per_data["column_ids"])
            row_ids.append(per_data["row_ids"])
            prev_label_ids.append(per_data["prev_label_ids"])
            column_ranks.append(per_data["column_ranks"])
            inv_column_ranks.append(per_data["inv_column_ranks"])
            numeric_relations.append(per_data["numeric_relations"])
            label_id.append(per_data["label_id"])
            position_id.append([i for i in range(512)])
            num_node = len(set(per_data["bert_maps"]))
            word_masks[index, :(per_data["input_ids"].index(102) + 1)] = 1
            ## for graph
            graph_1 = per_data["graph_edg1"]
            graph_2 = per_data["graph_edg2"]
            start_poss.append(per_data["start_pos"])
            end_poss.append(per_data["end_pos"])

            graph_1 = coo_matrix((graph_1["values"], (graph_1["row"], graph_1["col"])),
                                 shape=(num_node, num_node)).toarray()
            graph_2 = coo_matrix((graph_2["values"], (graph_2["row"], graph_2["col"])),
                                 shape=(num_node, num_node)).toarray()

            graph_mask[index, :num_node] = 1
            adj_matrix[index, 0, :num_node, :num_node] = graph_1
            adj_matrix[index, 1, :num_node, :num_node] = graph_2

            ## for bert
            bert_words.append(per_data["bert_words"])
            bert_maps.append(per_data["bert_maps"])

        max_len_bert = max([len(ex) for ex in bert_words])
        bert_words_ = np.zeros([batch_size, 512])
        bert_maps_ = np.zeros([batch_size, 512])
        start_pos = np.zeros([batch_size, 512])
        end_pos = np.zeros([batch_size, 512])
        for i in range(batch_size):
            bert_words_[i][:len(bert_words[i])] = bert_words[i]
            bert_maps_[i][:len(bert_maps[i])] = bert_maps[i]
            start_pos[i][:len(start_poss[i])] = start_poss[i]
            end_pos[i][:len(end_poss[i])] = end_poss[i]
        input_ids = torch.as_tensor(input_ids)
        input_mask = torch.as_tensor(input_mask)
        segment_ids = torch.as_tensor(segment_ids)
        column_ids = torch.as_tensor(column_ids)
        row_ids = torch.as_tensor(row_ids)
        prev_label_ids = torch.as_tensor(prev_label_ids)
        column_ranks = torch.as_tensor(column_ranks)
        inv_column_ranks = torch.as_tensor(inv_column_ranks)
        numeric_relations = torch.as_tensor(numeric_relations)
        label_id = torch.as_tensor(label_id)
        position_id = torch.as_tensor(position_id)
        adj_matrix = torch.as_tensor(adj_matrix)
        graph_mask = torch.as_tensor(graph_mask)
        bert_maps_ = torch.as_tensor(bert_maps_)
        bert_words_ = torch.as_tensor(bert_words_)
        word_masks = torch.as_tensor(word_masks)
        start_pos = torch.as_tensor(start_pos)
        end_pos = torch.as_tensor(end_pos)
        return input_ids, input_mask, segment_ids, column_ids, row_ids, prev_label_ids, column_ranks, \
               inv_column_ranks, numeric_relations, label_id, position_id, adj_matrix, graph_mask, bert_words_, bert_maps_, word_masks, start_pos, end_pos

class joint_dataset(Dataset):
    def __init__(self, features):
        self.data = features

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prepare_data(args, data_type):
    data_set = []
    with open(os.path.join(args.data_dir, '{}.pkl'.format(data_type)), 'rb') as f:
        data_dict = pickle.load(f)

    with open(os.path.join(args.data_dir, '{}_graph1.pkl'.format(data_type)), 'rb') as f:
        graph_1 = pickle.load(f)

    with open(os.path.join(args.data_dir, '{}_graph2.pkl'.format(data_type)), 'rb') as f:
        graph_2 = pickle.load(f)

    with open(os.path.join(args.data_dir, '{}_map.pkl'.format(data_type)), 'rb') as f:
        map = pickle.load(f)
    ## concate the bert data and the graph data
    for key, value in data_dict.items():
        start_pos = []
        end_pos = []
        index = 0
        early_node = 0
        data_dict[key]["graph_edg1"] = graph_1[key]
        data_dict[key]["graph_edg2"] = graph_2[key]

        bert_word = []
        bert_map = []
        for ind, node_name in enumerate(map[key]):
            for i in map[key][node_name]:
                bert_word.append(data_dict[key]["input_ids"][i])
                bert_map.append(ind)
            if ind == 0:
                start_pos.append(index)
                end_pos.append(index)
            else:
                if map[key][node_name] != early_node:
                    start_pos.append(index)
                    end_pos.append(index)
                else:
                    end_pos[-1] = index
                early_node = map[key][node_name]
            index += 1
        assert len(start_pos) == len(end_pos)
        assert len(set(bert_map)) == len(map[key].keys())
        data_dict[key]["bert_words"] = bert_word
        data_dict[key]["bert_maps"] = bert_map
        data_dict[key]["start_pos"] = start_pos
        data_dict[key]["end_pos"] = end_pos
        data_set.append(data_dict[key])
    collator = MyCollator(2)
    if data_type == "train":
        shuffle = True
        batch_size = args.train_batch_size
    else:
        shuffle = False
        batch_size = args.eval_batch_size

    dataset_ = joint_dataset(data_set)
    dataloader = DataLoader(dataset_, batch_size=batch_size, num_workers=1, shuffle=shuffle, collate_fn=collator)
    num_examples = len(map)
    return dataloader, num_examples



