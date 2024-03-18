import re


class Doc:
    def __init__(self, doc_id, sentence_list, nodes, child2refr_timex_edges, child2refr_event_edges):
        """
        :param text: a string
        """
        self.doc_id = doc_id
        self.sentence_list = sentence_list
        self.nodes = nodes
        self.child2refr_timex_edges = child2refr_timex_edges
        self.child2refr_event_edges = child2refr_event_edges
        self.edges = child2refr_timex_edges + child2refr_event_edges


class Node:
    def __init__(self, ID, node_type, text,
                 start_token_idx_in_doc, end_token_idx_in_doc,
                 note_idx, section_id=-1,
                 snt_index_in_doc=None, start_word_index_in_snt=None, end_word_index_in_snt=None,
                 node_id=None):
        self.ID = ID

        self.start_token_idx_in_doc = start_token_idx_in_doc
        self.end_token_idx_in_doc = end_token_idx_in_doc

        self.snt_index_in_doc = snt_index_in_doc
        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt

        self.note_idx = note_idx
        self.type = node_type
        self.text = text

        self.section_id = section_id
        self.node_id = node_id

    def __str__(self):
        return '\t'.join([self.ID, self.type, self.text])


def get_idx_in_doc(sent_id, sent_token_start, sent_list):
    s = 0
    if sent_id < 0:
        # Meta nodes
        return sent_token_start
    for s_idx, sent in enumerate(sent_list):
        if s_idx < sent_id:
            s += len(sent)
        else:
            break
    return s + sent_token_start


def get_a_node(entry, type, text, note_idx):
    """
    :param entry: e.g. 2_3_4,
    :param type: Timex or Event
    :param text: a list of sentences in this doc
    :param note_idx: filename or note id
    :return: a Node
    """
    sent_idx, sent_start, sent_end = [int(i) for i in entry.split("_")]
    if entry == "-1_-1_-1":
        nd = get_root_node(note_idx)
    elif entry == "-2_-2_-2":
        nd = get_padding_node(note_idx)
    else:
        nd_text = " ".join(text[sent_idx][sent_start:(sent_end + 1)])
        doc_start = get_idx_in_doc(sent_idx, sent_start, text)
        doc_end = get_idx_in_doc(sent_idx, sent_end, text)
        if entry.startswith("0_0_"):
            # Is DCT
            nd = Node(ID=entry,
                      node_type="DCT", text=nd_text,
                      start_token_idx_in_doc=doc_start, end_token_idx_in_doc=doc_end,
                      note_idx=note_idx,
                      snt_index_in_doc=sent_idx, start_word_index_in_snt=sent_start, end_word_index_in_snt=sent_end)
        else:
            assert sent_idx >= 0, entry
            assert type in ["Event", "Timex"], type
            nd = Node(ID=entry,
                      node_type=type, text=nd_text,
                      start_token_idx_in_doc=doc_start, end_token_idx_in_doc=doc_end,
                      note_idx=note_idx,
                      snt_index_in_doc=sent_idx, start_word_index_in_snt=sent_start, end_word_index_in_snt=sent_end)
    return nd


def get_root_node(note_idx):
    return Node(ID="-1_-1_-1", node_type="ROOT", text="<ROOT>",
                 start_token_idx_in_doc=-1, end_token_idx_in_doc=-1,
                 note_idx=note_idx, section_id=-1,
                 snt_index_in_doc=-1, start_word_index_in_snt=-1, end_word_index_in_snt=-1, node_id=-1)


def get_padding_node(note_idx):
    return Node(ID="-2_-2_-2", node_type="PAD", text="<PAD>",
                 start_token_idx_in_doc=-2, end_token_idx_in_doc=-2,
                 note_idx=note_idx, section_id=-2,
                 snt_index_in_doc=-2, start_word_index_in_snt=-2, end_word_index_in_snt=-2, node_id=-2)


def get_no_ref_event_node(note_idx):
    return Node(ID="-9_-9_-9", node_type="NO-REF-EVENT", text="<NO-REF-EVENT>",
                 start_token_idx_in_doc=-9, end_token_idx_in_doc=-9,
                 note_idx=note_idx, section_id=-9,
                 snt_index_in_doc=-9, start_word_index_in_snt=-9, end_word_index_in_snt=-9, node_id=-9)


class TDGRelation:
    def __init__(self, ID, rel_type, note_idx, source, target, label, doc_id=None):
        self.ID = ID
        self.type = rel_type
        self.note_idx = note_idx
        self.source = source
        self.target = target
        self.label = label
        self.doc_id = doc_id
        # From child to parent: child:parent
        self.pair_type = target.type.upper() + ":" + source.type.upper()

    def __str__(self):
        return '\t'.join([self.ID, self.label,
                          "Source: ",
                          '\t'.join([self.source.ID, self.source.type, self.source.text]),
                          "Target: ",
                          '\t'.join([self.target.ID, self.target.type, self.target.text]),
                          "Pair_type: ", self.pair_type
                          ])


def get_nd_start_token_idx(nd):
    return nd.start_token_idx_in_doc
