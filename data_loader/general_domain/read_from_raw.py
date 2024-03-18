"""
Read in text, gold edges, nodes from the raw data;
Specify random seed;
Distinguish no_ref_event vs. root
"""
import collections
import sys
sys.path.append(".")
# print(sys.path)
from data_loader.entity import *
# from entity import *

ref_timex_max_padded_candidate = 20
ref_event_max_padded_candidate = 20


def read_all_docs(input_file, list_candidates=False):
    """
    :param input_file: e.g. dev.txt
    :return: a list of docs
    """
    with open(input_file, "r", encoding="utf-8") as fr:
        lines = fr.read()

    all_docs = [doc for doc in lines.split("\n\n") if doc != "\n"]
    data = []

    # stat = collections.defaultdict(int)
    # stat_ee = collections.defaultdict(int)
    # stat_et = collections.defaultdict(int)
    # num_c = 0

    for doc in all_docs:
        filename, sent_and_edges = doc.split(":SNT_LIST")

        sentence_string, edge_string = sent_and_edges.split("EDGE_LIST")
        sentences = [sent.split(" ") for sent in sentence_string.split("\n") if len(sent) != 0]
        edges = edge_string.split("\n")
        edges = [edge.split("\t") for edge in edges if len(edge) != 0]

        nodes, to_timex_parent_edges, to_event_parent_edges = read_one_doc(sentences, edges, filename, list_candidates)
        nodes.sort(key=get_nd_start_token_idx)

        data.append(
            Doc(doc_id=filename,
                sentence_list=sentences,
                nodes=nodes,
                child2refr_timex_edges=to_timex_parent_edges,
                child2refr_event_edges=to_event_parent_edges))
    # print(sum(stat.values()), sum(stat.values())/len(stat))
    return data


def read_one_doc(sentence_list, edges, file_id, list_candidates=False):
    node_list = []
    # edge_list = []
    child2timex_parent_edge_list, child2event_parent_edge_list = [], []
    node_ids = set()
    node_list_no_dup = []
    for edge in edges:
        try:
            child, child_type, parent, _ = edge
        except ValueError:
            child, child_type = edge
            parent = '-1_-1_-1'
        assert child_type in ["Event", "Timex"]
        nd = get_a_node(child, child_type, sentence_list, file_id)
        node_list.append(nd)

        if child not in node_ids:
            node_list_no_dup.append(nd)
        node_ids.add(child)

    for n_idx, nd in enumerate(node_list_no_dup):
        nd.node_id = n_idx

    nd_id2nd = {nd.ID: nd for nd in node_list_no_dup}
    root_node = get_root_node(file_id)
    nd_id2nd["-1_-1_-1"] = root_node
    no_ref_event_node = root_node #get_no_ref_event_node(file_id)

    if not list_candidates:
        for edge in edges:
            try:
                child_entry, child_type, parent_entry, label = edge
            except ValueError:
                child_entry, child_type = edge
                parent_entry = '-1_-1_-1'
                label = "Depend-on"
            child_nd = nd_id2nd[child_entry]
            parent_nd = nd_id2nd[parent_entry]

            rel_tup = TDGRelation(ID=child_entry+"-"+parent_entry,
                                  rel_type=child_nd.type+"-"+parent_nd.type, note_idx=file_id,
                                  source=parent_nd, target=child_nd, label=label,
                                  doc_id=file_id)
            if child_nd.type == "Event" and parent_nd.type != "Event":
                child2event_parent_edge_list.append(rel_tup)
            else:
                child2timex_parent_edge_list.append(rel_tup)
            # edge_list.append(rel_tup)
    else:
        for edge in edges:
            try:
                child_entry, child_type, parent_entry, label = edge
            except ValueError:
                child_entry, child_type = edge
                parent_entry = '-1_-1_-1'
                label = "Depend-on"
            child_nd = nd_id2nd[child_entry]
            parent_nd = nd_id2nd[parent_entry]

            if child_nd.type != "Event":
                # Timex to reference timex edges
                one_child2timex_parent_edges = generate_cand_timex_parent(file_id,
                                           child=child_nd, candidates=node_list_no_dup,
                                           gold_parent=parent_nd, gold_label=label,
                                           root_node=root_node)
                child2timex_parent_edge_list.append(one_child2timex_parent_edges)
            else:
                if parent_nd.type == "Event" or parent_nd.ID == "-1_-1_-1":
                    # Event to reference event edges
                    one_child2event_parent_edges = generate_cand_event_parent(
                        file_id, child=child_nd, candidates=node_list_no_dup,
                        gold_parent=parent_nd, gold_label=label,
                        no_ref_event_node=no_ref_event_node)
                    child2event_parent_edge_list.append(one_child2event_parent_edges)
                else:
                    # Event to reference timex edges
                    assert parent_nd.type in ["Timex", "DCT"], parent_nd.type
                    one_child2timex_parent_edges = generate_cand_timex_parent(
                        file_id, child=child_nd, candidates=node_list_no_dup,
                        gold_parent=parent_nd, gold_label=label)
                    child2timex_parent_edge_list.append(one_child2timex_parent_edges)
    return node_list_no_dup, child2timex_parent_edge_list, child2event_parent_edge_list


def generate_cand_timex_parent(file_id, child, candidates, gold_parent, gold_label, root_node=None):
    cand_edges = []
    for cand_p in candidates:
        if cand_p.type == "Event":
            continue
        if cand_p.ID == child.ID:
            continue
        if cand_p.ID == gold_parent.ID:
            one_edge = get_a_tup(cand_p, child, gold_label, file_id)
        else:
            # assert cand_p.ID != gold_parent.ID
            one_edge = get_a_tup(cand_p, child, "NO_EDGE", file_id)
        cand_edges.append(one_edge)
    if root_node:
        if root_node.ID == gold_parent.ID:
            cand_edges.append(get_a_tup(root_node, child, gold_label, file_id))
        else:
            cand_edges.append(get_a_tup(root_node, child, "NO_EDGE", file_id))
    return cand_edges


def generate_cand_event_parent(file_id, child, candidates, gold_parent, gold_label, no_ref_event_node):
    cand_edges = []
    assert child.type == "Event"
    if gold_parent.ID == "-1_-1_-1":
        cand_edges.append(get_a_tup(no_ref_event_node, child, gold_label, file_id))
    else:
        cand_edges.append(get_a_tup(no_ref_event_node, child, "NO_EDGE", file_id))
    for cand_p in candidates:
        if cand_p.type != "Event":
            continue
        if cand_p.ID == child.ID:
            continue
        # Has already added no_ref_event_node
        if cand_p.ID == "-1_-1_-1":
            continue

        if cand_p.snt_index_in_doc - child.snt_index_in_doc > 2:
            continue

        if cand_p.ID == gold_parent.ID:
            one_edge = get_a_tup(cand_p, child, gold_label, file_id)
        else:
            one_edge = get_a_tup(cand_p, child, "NO_EDGE", file_id)
        cand_edges.append(one_edge)
    return cand_edges


def get_a_tup(parent, child, label, file_id):
    return TDGRelation(ID=child.ID + "-" + parent.ID,
                       rel_type=child.type + "-" + parent.type, note_idx=file_id,
                       source=parent, target=child, label=label,
                       doc_id=file_id)
