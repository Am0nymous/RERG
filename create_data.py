import logging
import os.path
import time
import argparse
from tapas.utils import tabfact_utils
from tapas.utils import number_annotation_utils
from tapas.utils import tf_example_utils
from tqdm import tqdm

import pickle
import spacy
nlp = spacy.load('en')
def parse_opt():
    parser = argparse.ArgumentParser()

    ## set dir
    parser.add_argument('-home', default=".", type=str)
    parser.add_argument('-input_dir', default="./TABFACT", type=str)
    parser.add_argument('-output_dir', default="./TABFACT_tapas_data", type=str)
    parser.add_argument('-log_dir', default="./Running_info", type=str)
    parser.add_argument('-vocab_file', default="convert_model/tapas_model/vocab.txt", type=str)
    parser.add_argument('-max_seq_length', default=512, type=int)
    parser.add_argument('-MAX_TABLE_ID', default=512, type=int)
    args = parser.parse_args()
    return args

def find(a, b):
    index = []
    for i, it in enumerate(b):
        if a == it:
            index.append(i)

class edge_dict:
    edge = {}

def state2table_ed(origin2_map, map_dict_new, input_ids, map_dict, border, node_map, dep_map1, dep_map2, dep_map3, dep_edg):
    ## 1 indicates the relation between statement and table
    ## 2 indicates the relation among the same row
    ## 3 indicates the relation among the same column
    # size (node, node) the value are the one of the "1, 2, 3"
    type1_ed = []
    node = []
    node_name = node_map.keys()
    for i, key in enumerate(node_map):
        if i == 0:
            node_ = [1 for i in range(len(node_name))]
            node.append(node_)
            continue
        node_ = [0 for i in range(len(node_name))]
        if isinstance(key, int):   #### for statement
            node__ = set()
            for ii in node_map[key]:## obtain state node <---> table node
                for j, it in enumerate(input_ids):
                    if j <= border:
                        continue
                    if input_ids[ii] == input_ids[j]:
                        node__.add(origin2_map[j])
            node__ = list(node__)
            for i in node__:
                node_[i] = 1
    
            if key in dep_map2.keys():  ## obtain state node <---> state node (spacy dependency parsing)
                no = dep_map3[dep_map1[dep_map2[key]]]
                edg = dep_edg[dep_map2[key]]
                for i in no:
  
                    node_[origin2_map[i]] = edg
        else:    ### for table elements
            node__ = set()
            for ii in node_map[key]:
                for j, it in enumerate(input_ids):
                    if j <= border:
                        if input_ids[ii] == input_ids[j]:
                            node__.add(origin2_map[j])
            node__ = list(node__)
            for i in node__:
                node_[i] = 1
            f_row = key[0]
            f_column = key[1]
            for i, key in enumerate(node_name):
                if isinstance(key, tuple):
                    if f_row == key[0]:
                        node_[i] = 2
                    if f_column == key[1]:
                        node_[i] = 3

        node.append(node_)
    row = []
    col = []
    values = []
    num_node = len(node_name)
    type_1 = {}
    for i in range(num_node):
        for j in range(num_node):
            if node[i][j] != 0:
                if node[j][i] == 0:
                    row.append(j)
                    col.append(i)
                    values.append(node[i][j])
                row.append(i)
                col.append(j)
                values.append(node[i][j])
    type_1['row'] = row
    type_1['col'] = col
    type_1['values'] = values

    return type_1


def number_ed_b(origin2_map, rank_dict, relation_dict, map_dict, border, node_map):
    # the number_ed indicates the number relation between nodes, specifically,
    # one is the statement and table, the other is the cells in the same column.
    # the one relation is the corresponding value in relation_dict, the other relation is
    # "bigger than" 5 or "small  than" 7 or "equal "9"
    type1_ed = []
    node = []
    indexs = relation_dict[1]
    relation_dict = relation_dict[0]
    node_name = node_map.keys()
    for i, key in enumerate(node_map):
        node_ = [0 for i in range(len(node_name))]
        if isinstance(key, int):  #### for statement
            for o_1, j in enumerate(indexs):
                for ii in j:
                    node_id = origin2_map[ii]
                    if node_id == i:
                        for i_, value in origin2_map.items():
                            node_[origin2_map[i_]] = relation_dict[o_1][i_]
        else:  ### for table columns
            f_column = key[1]
            value_column = rank_dict[key]
            orders = []
            for i, key in enumerate(rank_dict):
                if isinstance(key, tuple):
                   if f_column == key[1]:
                       orders.append(rank_dict[key])
            if sum(orders) !=0:
                for i, key in enumerate(rank_dict):
                    if isinstance(key, tuple):
                        if f_column == key[1]:
                            if value_column >= rank_dict[key]:
                                node_[i] = 1

        node.append(node_)

    row = []
    col = []
    values = []
    num_node = len(node_name)
    type_1 = {}
    for i in range(num_node):
        for j in range(num_node):
            if node[i][j] != 0:
                if node[j][i] == 0:
                    row.append(j)
                    col.append(i)
                    values.append(1)
                row.append(i)
                col.append(j)
                values.append(1)
    type_1['row'] = row
    type_1['col'] = col
    type_1['values'] = values

    return type_1


def number_ed(origin2_map, rank_dict, relation_dict, map_dict, border, node_map):
    # the number_ed indicates the number relation between nodes, specifically,
    # one is the statement and table, the other is the cells in the same column.
    # the one relation is the corresponding value in relation_dict, the other relation is
    # "bigger than" 5 or "small  than" 7 or "equal "9"
    type1_ed = []
    node = []
    indexs = relation_dict[1]
    relation_dict = relation_dict[0]
    node_name = node_map.keys()
    for i, key in enumerate(node_map):
        node_ = [0 for i in range(len(node_name))]
        if isinstance(key, int):  #### for statement
            for o_1, j in enumerate(indexs):
                for ii in j:
                    node_id = origin2_map[ii]
                    if node_id == i:
                        for i_, value in origin2_map.items():
                            node_[origin2_map[i_]] = relation_dict[o_1][i_]
        else:  ### for table columns
            f_column = key[1]
            value_column = rank_dict[key]
            orders = []
            for i, key in enumerate(rank_dict):
                if isinstance(key, tuple):
                   if f_column == key[1]:
                       orders.append(rank_dict[key])
            if sum(orders) !=0:
                for i, key in enumerate(rank_dict):
                    if isinstance(key, tuple):
                        if f_column == key[1]:
                            if value_column > rank_dict[key]:
                                node_[i] = 9
                            elif value_column < rank_dict[key]:
                                node_[i] = 8
                            elif value_column == rank_dict[key]:
                                node_[i] = 7
        node.append(node_)

    row = []
    col = []
    values = []
    num_node = len(node_name)
    type_1 = {}
    for i in range(num_node):
        for j in range(num_node):
            if node[i][j] != 0:
                if node[j][i] == 0:
                    row.append(j)
                    col.append(i)
                    values.append((node[i][j]))
                row.append(i)
                col.append(j)
                values.append(node[i][j])
    type_1['row'] = row
    type_1['col'] = col
    type_1['values'] = values

    return type_1


def main():
    args = parse_opt()

    ## set logger print to console and log file
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(args.log_dir + '/Logs/')
    log_name = log_path + rq + '.log'
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("begining create data...")

    tables = tabfact_utils._convert_tables(args.input_dir)
    questions = tabfact_utils.read_questions(args.input_dir)
    logger.info("total tables are: {}".format(len(tables)))
    logger.info("total statements are: {}".format(len(questions)))
    splits = {
        'train': 'train_id.json',
        'test': 'test_id.json',
        'dev': 'val_id.json',
         'simple_test': 'simple_test_id.json',
         'small_test':'small_test_id.json',
         'complex_test':'complex_test_id.json',
    }

    label_dict = {}

    for splits, file in splits.items():
        logger.info("processing {} data".format(splits))
        graph_type1 = {}
        graph_type2 = {}
        map = {}
        exampless = {}
        examples = []
        num_questions = 0
        num_conversion_errors = 0
        splist_data = tabfact_utils._convert_data(questions, os.path.join(args.input_dir, 'data', file), tables)

        config = tf_example_utils.ClassifierConversionConfig(
            vocab_file=args.vocab_file,
            max_seq_length=args.max_seq_length,
            max_column_id=args.MAX_TABLE_ID,
            max_row_id=args.MAX_TABLE_ID,
            strip_column_names=False,
            add_aggregation_candidates=False,
        )
        converter = tf_example_utils.ToClassifierTensorflowExample(config)

        for interaction in tqdm(splist_data):
            number_annotation_utils.add_numeric_values(interaction)
            for i in range(len(interaction.questions)):
                num_questions += 1
                try:
                    tmp, relation_graph= converter.convert(interaction, i)

                    examples.append(tmp)
                    tmp_dict = tmp.features.feature._values
                    model_required = {}
                    id = tmp_dict["question_id"].bytes_list.value[0].decode('utf-8')
                    model_required["input_ids"] = tmp_dict["input_ids"].int64_list.value._values
                    model_required["input_mask"] = tmp_dict["input_mask"].int64_list.value._values
                    model_required["segment_ids"] = tmp_dict["segment_ids"].int64_list.value._values
                    model_required["column_ids"] = tmp_dict["column_ids"].int64_list.value._values
                    model_required["row_ids"] = tmp_dict["row_ids"].int64_list.value._values
                    model_required["prev_label_ids"] = tmp_dict["prev_label_ids"].int64_list.value._values
                    model_required["column_ranks"] = tmp_dict["column_ranks"].int64_list.value._values
                    model_required["inv_column_ranks"] = tmp_dict["inv_column_ranks"].int64_list.value._values
                    model_required["numeric_relations"] = tmp_dict["numeric_relations"].int64_list.value._values
                    model_required["label_id"] = tmp_dict["classification_class_index"].int64_list.value._values

                    tokens = [tf_example_utils.Token('[CLS]','[CLS]')] + converter._tokenizer.tokenize(interaction.questions[i].original_text) \
                             + [tf_example_utils.Token('[SEP]','[SEP]')]

                    doc = nlp(interaction.questions[i].original_text)
                    dep_map1 = {}
                    dep_map2 = {}
                    dep_map3 = {}
                    dep_edg = {}
                    for i, it in enumerate(doc):
                        tmp = []
                        dep_map1[i] = list(doc).index(it.head)
                        if it.dep_  not in label_dict.keys():
                            dep_edg[i] = 30 + len(label_dict.keys())
                            label_dict[str(it.dep_)] = dep_edg[i]
                        else:
                            dep_edg[i] = label_dict[str(it.dep_)]
                        for j, iq in enumerate(tokens):
                            if it.text == iq.original_text:
                                tmp.append(j)
                                dep_map2[j] = i
                        dep_map3[i] = tmp

                    map_dict = {}
                    for i, t in enumerate(tokens):
                        if i==0:
                            map_dict[i] = [0]
                            continue
                        in_value = map_dict[i-1][-1] + 1
                        inner = []
                        span_len = len(converter._to_token_ids([t]))
                        for off_set in range(span_len):
                            in_value_ = in_value + off_set
                            inner.append(in_value_)
                        map_dict[i] = inner
                    # in_value indicates the end of the statement
                    if in_value_ != model_required['segment_ids'].index(1) - 1:
                        print(in_value_, model_required['segment_ids'].index(1) - 1, map_dict, tokens)
                    assert in_value_ == model_required['segment_ids'].index(1) - 1

                    input_ids = model_required['input_ids']
                    rows = model_required['row_ids']
                    columns = model_required['column_ids']
                    ranks = model_required['column_ranks']
                    inv_ranks = model_required['inv_column_ranks']
                    inner = []
                    for i, t in enumerate(model_required['input_ids']):
                        if i <= in_value_:
                            continue
                        if columns[i] ==0:
                            continue
                        if i < len(rows) -1 and columns[i]!= columns[i+1]:
                            inner.append(i)
                            map_dict[(rows[i], columns[i])] = inner
                            inner = []
                        else:
                            inner.append(i)
                            if i ==len(rows) -1:
                                map_dict[(rows[i], columns[i])] = inner



                    node_map = {}
                    number = list(set([l for L in relation_graph[1] for l in L]))

                    for i, key in enumerate(map_dict):
                        if isinstance(key, int):  #### for statement
                            for ii in map_dict[key]:
                                for j, it in enumerate(input_ids):
                                    if j <= in_value_:
                                        continue
                                    if input_ids[ii] == input_ids[j]:
                                      node_map[key] = map_dict[key]
                            if i in number:
                                node_map[key] = map_dict[key]

                        if isinstance(key, tuple):
                            node_map[key] = map_dict[key]
                    node_map = map_dict.copy()
                    origin2_map = {}
                    map_dict_new = {}
                    for index, key in enumerate(node_map):
                        map_dict_new[index] = node_map[key]
                    for key, value in map_dict_new.items():
                        for it in value:
                            origin2_map[it] = key

                    # rank, inv_rank, relation
                    rank_dict = {}
                    inv_rank_dict = {}
                    node_map_ = {}
                    for key, value in node_map.items():
                        if isinstance(key, int):
                            rank_dict[key] = -1
                            inv_rank_dict[key] = -1

                        if isinstance(key, tuple):
                            rank_dict[key] = ranks[node_map[key][0]]
                            inv_rank_dict[key] = inv_ranks[map_dict[key][0]]

                    for key, value in node_map.items():
                        if isinstance(key, tuple):
                            key = str(key)
                            node_map_[key] = value
                        else:
                            node_map_[key] = value

                    type_1 = state2table_ed(origin2_map, map_dict_new, input_ids, map_dict, in_value_, node_map, dep_map1, dep_map2, dep_map3, dep_edg)
                    type_2 = number_ed(origin2_map, rank_dict, relation_graph, map_dict, in_value_, node_map)


                    model_required["segment_ids"] = tmp_dict["segment_ids"].int64_list.value._values
                    exampless[id] = model_required
                    graph_type1[id] = type_1
                    graph_type2[id] = type_2

                    map[id] = node_map_


                except ValueError as e:
                    num_conversion_errors += 1
                    logging.info("Can't convert interaction: %s error: %s", interaction.id, e)


        logger.info(f'Processed: {splits}')
        logger.info(f'Num questions processed: {num_questions}')
        logger.info(f'Num examples: {len(examples)}')
        logger.info(f'Num conversion errors: {num_conversion_errors}')

        with open("./TABFACT_tapas_data/{}.pkl".format(splits), 'wb') as f:
            pickle.dump(exampless, f)

        with open("./TABFACT_tapas_data/{}_graph1.pkl".format(splits), 'wb') as f:
            pickle.dump(graph_type1, f)

        with open("./TABFACT_tapas_data/{}_graph2.pkl".format(splits), 'wb') as f:
            pickle.dump(graph_type2, f)

        with open("./TABFACT_tapas_data/{}_map.pkl".format(splits), 'wb') as f:
            pickle.dump(map, f)


    pass


if __name__ == "__main__":
    main()
