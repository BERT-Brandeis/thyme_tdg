from collections import defaultdict
from pprint import pprint
from typing import Dict, List
from dataclasses import dataclass
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from collections import defaultdict
import sys

sys.path.append(".")
from data_loader.thyme_tdg.data_preparation_thyme_sent import make_test_data
from data_loader.thyme_tdg.to_nli import generate_test_example
from data_loader.util import THYME_TDG_LABELS, THYME_TDG_LABEL_TEMPLATES, THYME_TDG_VALID_CONDITIONS
from nli_parser import *


def parse_one_doc(orig_edges, input_features, model, label_template, template2label):
    # edges: all_input_features; pred_edges: cand_p
    # model: clf
    pred_all_edges = []
    for c_id, child in enumerate(input_features):
        pred_edges = model.decode_tdg(child, multiclass=True)
        pred_parent_idx = int(pred_edges / len(label_template))
        pred_label = pred_edges % len(label_template)
        assert orig_edges[c_id][pred_parent_idx][1].ID == child[0].subj_child_id
        pred_one_edge = [child[0].subj_child_id, orig_edges[c_id][pred_parent_idx][1].type,
                         orig_edges[c_id][pred_parent_idx][0].ID, template2label[pred_label]]
        if orig_edges[c_id][pred_parent_idx][0].ID == "-1_-1_-1" and orig_edges[c_id][pred_parent_idx][1].type == "EVENT":
            continue
        pred_all_edges.append(pred_one_edge)
    return pred_all_edges


def main(args):
    template_mapping = THYME_TDG_LABEL_TEMPLATES
    rules = THYME_TDG_VALID_CONDITIONS

    testing_data, doc_id_list = make_test_data(args.input_file)
    # testing_data = read_all_docs(args.input_file, list_candidates=True)

    clf = NLIRelationClassifierWithMappingHead(
        labels=THYME_TDG_LABELS,
        template_mapping=template_mapping,
        pretrained_model=args.model_dir,
        valid_conditions=rules
    )

    positive_templates = defaultdict(list)

    all_input_features = []
    all_input_edges = []
    for doc in testing_data:
        snt_list, testing_example_list, node_list = doc
        one_doc_features = []
        one_doc_edges = []
        for child in testing_example_list:
            input_features = []
            input_edges = []
            for cand_p in child:
                # Skip padding edges
                if cand_p[0].ID == "-2_-2_-2":
                    continue
                one_test_instance = generate_test_example(cand_p, snt_list, has_dist_feature=True)
                input_features.append(one_test_instance)
                input_edges.append(cand_p)
            one_doc_features.append(input_features)
            one_doc_edges.append(input_edges)
        all_input_features.append(one_doc_features)
        all_input_edges.append(one_doc_edges)

    template_label_lst = []
    template2label = {}
    temp_idx = 0
    for l, temps in template_mapping.items():
        for temp in temps:
            template_label_lst.append(temp)
            template2label[temp_idx] = l
            temp_idx += 1

    to_write = ""
    for d_idx, doc_input_feature in enumerate(all_input_features):
        edges_in_this_doc = all_input_edges[d_idx]
        doc_id = edges_in_this_doc[0][0][1].note_idx
        to_write += (doc_id + ":SNT_LIST\n")
        to_write += "EDGE_LIST\n"
        one_doc_parsed = parse_one_doc(edges_in_this_doc, doc_input_feature, clf, template_label_lst, template2label)
        for parsed_edge in one_doc_parsed:
            to_write += "\t".join(parsed_edge)
            to_write += "\n"
        if d_idx != len(all_input_features) - 1:
            to_write += "\n"
    with open(os.path.join(args.output_dir, args.output_file), "w", encoding="utf-8") as fw:
        fw.write(to_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()
    main(args)


