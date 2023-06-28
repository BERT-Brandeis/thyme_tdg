UNK_word = '<UNK>'
UNK_label = '<UNK>'
PAD_word = 'PADDING'
PAD_label = '<PAD>'
ROOT_word = 'ROOT'
ROOT_label = '<ROOT>'
AUTHOR_word = 'AUTHOR'
AUTHOR_label = '<AUTHOR>'
NO_REF_EVENT_word = 'NULL'
NO_REF_EVENT_label = '<NULL>'
DCT_word = 'DCT'
DCT_label = '<DCT>'


THYME_EDGE_LABEL_LIST = ['AFTER',  'OVERLAP', 'CONTAINS-SUBEVENT', 'BEFORE', 'CONTAINS-SUBEVENT-INV',
                         'NOTED-ON-INV', 'AFTER/OVERLAP', 'CONTAINS', 'Depend-on', 'BEGINS-ON', 'ENDS-ON',
                         'CONTAINS-INV', 'NOTED-ON']

TEMPORAL_EDGE_LABEL_LIST = [
'before',
'after',
'overlap',
'included',
'Depend-on']


EVENT_TIMEX_BIO2id = {'b_e':0, 'i_e':1, 'b_t':2, 'i_t':3, 'o':4}
id2EVENT_TIMEX_BIO = {v: k for k, v in EVENT_CONC_BIO2id.items()}
EVENT_TIMEX_BIO = ['b_e', 'i_e', 'b_t', 'i_t', 'o']

class Node:
    def __init__(self, snt_index_in_doc=-1, start_word_index_in_snt=-1, end_word_index_in_snt=-1, node_index_in_doc=-1,
                 start_word_index_in_doc=-1, end_word_index_in_doc=-1, words=ROOT_word, label=ROOT_label,
                 thyme_id=-1, note_idx=None, end_snt_index_in_doc=None):
        self.snt_index_in_doc = snt_index_in_doc

        if end_snt_index_in_doc is not None:
            self.end_snt_index_in_doc = end_snt_index_in_doc
        else:
            self.end_snt_index_in_doc = self.snt_index_in_doc

        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt
        self.node_index_in_doc = node_index_in_doc
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc

        self.words = words
        self.text = self.words
        self.label = label
        self.type = self.label

        self.thyme_id = thyme_id
        self.note_idx = note_idx

        self.is_DCT = ("DOCTIME" in self.label)

        self.short_ID = "_".join([str(snt_index_in_doc), str(start_word_index_in_snt), str(end_word_index_in_snt)])
        self.ID = '_'.join([str(thyme_id),
                            str(start_word_index_in_doc), str(end_word_index_in_doc),
                            str(snt_index_in_doc), str(self.end_snt_index_in_doc),
                            str(start_word_index_in_snt), str(end_word_index_in_snt)])

    def __str__(self):
        return '\t'.join([self.ID, self.words, self.label])

    def __eq__(self, other):
        if isinstance(other, Node):
            # Note that this does not check whether the two nodes are from the same document
            return self.ID == other.ID
        return False

    def __hash__(self):
        return hash(self.ID)


def get_root_node():
    return Node()


def get_no_ref_event_node():
    return Node(-5, -5, -5, -5, -5, -5, NO_REF_EVENT_word, NO_REF_EVENT_label, -5)


def get_padding_node():
    return Node(-2, -2, -2, -2, -2, -2, PAD_word, PAD_label, -2)


def get_dct_node():
    return Node(-7, -7, -7, -7, -7, -7, DCT_word, DCT_label, -7)


def is_root_node(node):
    return node.snt_index_in_doc == -1


def is_padding_node(node):
    return node.snt_index_in_doc == -2


def is_no_ref_event_node(node):
    return node.snt_index_in_doc == -5


def is_dct_node(node):
    return node.snt_index_in_doc == -7
