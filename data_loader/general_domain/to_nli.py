from collections import defaultdict
import random
import json
import string
import argparse
import sys
sys.path.append(".")
from data_loader.general_domain.read_from_raw import read_all_docs
from data_loader.util import *


class NLIInputFeature:
    def __init__(self, premise, hypothesis=None, nli_label=None, subj_child=None, obj_parent=None, temporal_label=None,
                 subj_child_id=None, obj_parent_id=None, doc_id=None, pair_type=None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.nli_label = nli_label

        self.label = nli_label

        # Assuming the relation goes from child to parent
        self.subj_child = subj_child
        self.obj_parent = obj_parent
        self.temporal_label = temporal_label

        self.subj_child_id = subj_child_id
        self.obj_parent_id = obj_parent_id
        self.doc_id = doc_id

        self.pair_type = pair_type

    def __str__(self):
        return '\t'.join(["Premise: ", self.premise, "Child: ", self.subj_child_id, "Parent: ", self.obj_parent_id])


def generate_test_example(instance, doc_sent_list, has_dist_feature):
    one_example = NLIInputFeature(
        premise=get_premise(instance.target, instance.source, doc_sent_list, has_dist_feature),
        # subject is child (target), object is parent (source)
        subj_child=instance.target.text,
        obj_parent=instance.source.text,
        temporal_label=instance.label,
        subj_child_id=instance.target.ID,
        obj_parent_id=instance.source.ID,
        doc_id=instance.doc_id,
        pair_type=instance.target.type.upper() + ":" + instance.source.type.upper()
    )
    return one_example


def generate_positive_example(instance, doc_sent_list, templates, posn, has_dist_feature):
    positive_templates = templates[instance.label]
    sampled_examples = []
    for template in random.sample(positive_templates, k=min(posn, len(positive_templates))):
        example = NLIInputFeature(
            premise=get_premise(instance.target, instance.source, doc_sent_list, has_dist_feature),
            # subject is child (target), object is parent (source)
            hypothesis=temporal_tuple_to_hypothesis(template=template,
                                                    subj_text=instance.target.text, obj_text=instance.source.text),
            nli_label=labels2id["entailment"],
            subj_child=instance.target.text,
            obj_parent=instance.source.text,
            temporal_label=instance.label,
            subj_child_id=instance.target.ID,
            obj_parent_id=instance.source.ID,
            doc_id=instance.doc_id
        )
        sampled_examples.append(example)
    return sampled_examples


def generate_neutral_example(instance, doc_sent_list, templates, pair_type2label, negn, has_dist_feature):
    possible_fake_label = list(set(pair_type2label[instance.pair_type]) - set([instance.label]))
    if not len(possible_fake_label):
        return []
    neutral_templates = []
    for label in possible_fake_label:
        neutral_templates.extend(templates[label])

    sampled_examples = []
    for template in random.sample(neutral_templates, k=min(negn, len(neutral_templates))):
        example = NLIInputFeature(
            premise=get_premise(instance.target, instance.source, doc_sent_list, has_dist_feature),
            # subject is child (target), object is parent (source)
            hypothesis=temporal_tuple_to_hypothesis(template=template,
                                                    subj_text=instance.target.text, obj_text=instance.source.text),
            nli_label=labels2id["neutral"],
            subj_child=instance.target.text,
            obj_parent=instance.source.text,
            temporal_label=instance.label,
            subj_child_id=instance.target.ID,
            obj_parent_id=instance.source.ID,
            doc_id=instance.doc_id
        )
        sampled_examples.append(example)
    return sampled_examples


def generate_negative_example(instance, doc_sent_list, templates, pair_type2label, negn, has_dist_feature):
    posible_fake_label = list(set(pair_type2label[instance.pair_type]) - set([instance.label]))
    if not len(posible_fake_label):
        return []
    negative_templates = []
    for label in posible_fake_label:
        negative_templates.extend(templates[label])

    sampled_examples = []
    for template in random.sample(negative_templates, k=min(negn, len(negative_templates))):
        example = NLIInputFeature(
            premise=get_premise(instance.target, instance.source, doc_sent_list, has_dist_feature),
            # subject is child (target), object is parent (source)
            hypothesis=temporal_tuple_to_hypothesis(template=template,
                                                    subj_text=instance.target.text, obj_text=instance.source.text),
            nli_label=labels2id["contradiction"],
            subj_child=instance.target.text,
            obj_parent=instance.source.text,
            temporal_label=instance.label,
            subj_child_id=instance.target.ID,
            obj_parent_id=instance.source.ID,
            doc_id=instance.doc_id
        )
        sampled_examples.append(example)
    return sampled_examples


def get_nli_for_one_instance(instance_wt_gold_parent, cand_p, doc_sent_list, templates, valid_condition, posn, negn, has_dis_feat):
    all_nli_examples = []
    entail_examples = generate_positive_example(instance_wt_gold_parent, doc_sent_list, templates, posn, has_dis_feat)
    neutral_examples = generate_neutral_example(instance_wt_gold_parent, doc_sent_list, templates, valid_condition,
                                                negn, has_dis_feat)
    for item in entail_examples:
        all_nli_examples.append(item)

    for item in neutral_examples:
        all_nli_examples.append(item)

    contradiction_examples = None
    for p in cand_p:
        contradiction_examples = generate_negative_example(p, doc_sent_list, templates, valid_condition,
                                                           negn, has_dis_feat)
        for item in contradiction_examples:
            all_nli_examples.append(item)
    return all_nli_examples


def get_pair_dis_feature(ent1, ent2):
    if ent1.snt_index_in_doc == ent2.snt_index_in_doc:
        sent_dis = " Same sentence ."
    elif ent2.snt_index_in_doc == -1:
        if ent1.type == "Event":
            sent_dis = " No reference event ."
        else:
            sent_dis = " Parent is Root ."
    elif ent2.ID.startswith("0_0_"):
        sent_dis = " Parent is DCT ."
    elif ent1.snt_index_in_doc < ent2.snt_index_in_doc:
        sent_dis = " Parent sentence after child sentence ."
    else:
        assert ent1.snt_index_in_doc > ent2.snt_index_in_doc
        sent_dis = " Parent sentence before child sentence ."

    if ent2.node_id < 0:
        node_dis = ""
    elif ent2.ID.startswith("0_0_"):
        node_dis = ""
    elif ent2.node_id - ent1.node_id == -1:
        node_dis = " Parent is the immediately previous node of the child node . "
    elif ent2.node_id - ent1.node_id == -2:
        node_dis = " Parent is two nodes before the child node in text order . "
    elif ent2.node_id - ent1.node_id == 1:
        node_dis = " Parent is the immediately succeeding node of the child node . "
    elif ent2.node_id > ent1.node_id:
        node_dis = " Parent node after the child node in text order . "
    else:
        node_dis = ""
    return node_dis, sent_dis


def get_premise(ent1, ent2, sent_list, has_dist_feature):
    # ent1: instance.target, ent2: instance.source
    sent_id1 = ent1.snt_index_in_doc
    sent_id2 = ent2.snt_index_in_doc

    if ent1.snt_index_in_doc < 0:
        s1 = ent1.text
    else:
        s1 = " ".join(sent_list[sent_id1])
    if ent2.snt_index_in_doc < 0:
        s2 = ent2.text
    else:
        s2 = " ".join(sent_list[sent_id2])

    if s1[-1] not in string.punctuation:
        s1 += " ."
    if s2[-1] not in string.punctuation:
        s2 += " ."

    if has_dist_feature:
        node_dis, sent_dis = get_pair_dis_feature(ent1, ent2)
    else:
        node_dis = ""
        sent_dis = ""

    if ent2.type not in ["Event", "ROOT", "NO-REF-EVENT"]:
        node_dis = ""

    if sent_id1 == sent_id2:
        return s1 + node_dis + sent_dis
    elif sent_id1 < sent_id2:
        return s1 + " " + s2 + node_dis + sent_dis
    else:
        if sent_id2 == -1:
            return s2 + " " + s1 + node_dis + sent_dis
        elif ent2.ID.startswith("0_0_"):
            return s2 + " " + s1 + node_dis + sent_dis
        else:
            return s2 + " " + s1 + node_dis + sent_dis


def temporal_tuple_to_hypothesis(template, subj_text, obj_text):
    if first_token_is_weekday(obj_text) or is_complete_month_date_year(obj_text):
        if TEMPLATE_TO_LABEL[template] == "included" and template == "{child} happened {parent}":
            obj_text = "on " + obj_text
        return f"{template.format(child=subj_text, parent=obj_text)} ."
    if first_token_is_year(obj_text) or month_only(obj_text):
        if TEMPLATE_TO_LABEL[template] == "included" and template == "{child} happened {parent}":
            obj_text = "in " + obj_text
        return f"{template.format(child=subj_text, parent=obj_text)} ."
    return f"{template.format(child=subj_text, parent=obj_text)} ."


def split_candidate_parents(candidate_parents):
    gold = []
    other_cand = []
    for edge in candidate_parents:
        if edge.label != "NO_EDGE":
            assert edge.label in TDG_LABELS
            gold.append(edge)
        else:
            other_cand.append(edge)
    if len(gold) == 0:
        return None, other_cand
    else:
        assert len(gold) == len(set(gold)) == 1, (len(gold), len(set(gold)))
        return gold[-1], other_cand


def convert_training_data(args):
    training_data = read_all_docs(args.input_file, list_candidates=True)

    pair_type2label = defaultdict(list)
    for label, pair_type in TDG_VALID_CONDITIONS.items():
        for pr in pair_type:
            pair_type2label[pr].append(label)

    nli_data = []
    nli_data_dict = []
    for doc in training_data:
        for child in doc.edges:
            gold_parent, list_of_cand_parents = split_candidate_parents(child)
            if not gold_parent:
                print("No gold parent: ")
                continue
            sampled_cand_parent = random.sample(list_of_cand_parents, k=min(3, len(list_of_cand_parents)))
            one_instance_nli_examples = get_nli_for_one_instance(gold_parent, sampled_cand_parent, doc.sentence_list,
                                                                 templates=TDG_LABEL_TEMPLATES,
                                                                 valid_condition=pair_type2label,
                                                                 posn=args.num_positive_example,
                                                                 negn=args.num_negative_example,
                                                                 has_dis_feat=args.dist_feature)
            nli_data.extend(one_instance_nli_examples)
    for item in nli_data:
        nli_data_dict.append(item.__dict__)
    # print(len(nli_data_dict))

    with open(args.output_file, "wt") as f:
        for example in nli_data:
            f.write(f"{json.dumps(example.__dict__)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_sampling_seed', type=int, default=42)
    parser.add_argument('--data_sampling_seed', type=int, default=52)
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    parser.add_argument("--dist_feature", action="store_true", default=False)
    parser.add_argument("--num_positive_example", default=3, type=int)
    parser.add_argument("--num_negative_example", default=2, type=int)

    args = parser.parse_args()

    random.seed(args.data_sampling_seed)
    convert_training_data(args)
