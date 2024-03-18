import re


labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}
TDG_LABELS = ['before', 'after', 'overlap', 'included', 'Depend-on']
TDG_LABELS = TDG_LABELS #  ["NO_EDGE"] +

THYME_TDG_LABELS = ['AFTER',  'OVERLAP', 'CONTAINS-SUBEVENT', 'BEFORE', 'CONTAINS-SUBEVENT-INV',
                    'NOTED-ON-INV', 'AFTER/OVERLAP', 'CONTAINS', 'Depend-on', 'BEGINS-ON', 'ENDS-ON',
                    'CONTAINS-INV', 'NOTED-ON']

TDG_LABEL_TEMPLATES = {
    # "NO_EDGE": ["{child} and {parent} are not related"],
    "before": ["{child} happened before {parent}"],
    "after": ["{child} happened after {parent}"],
    "overlap": ["{child} happened at around the same time as {parent}"],
    "included": ["{child} happened {parent}"],
    "Depend-on": ["{child} depended on {parent}"]
}
# In thyme-tdg, the relation goes from the parent to the child.
THYME_TDG_LABEL_TEMPLATES = {
    "AFTER": ["{parent} happened after {child}"],
    "OVERLAP": ["{parent} happened at around the same time as {child}"],
    "CONTAINS-SUBEVENT": ["{child} is a sub-event of {parent}"],
    "BEFORE": ["{parent} happened before {child}"],
    "CONTAINS-SUBEVENT-INV": ["{parent} is a sub-event of {child}"],

    # child is the result, parent is the test
    "NOTED-ON-INV": ["The {parent} test showed the result {child}"],##, "{child} showed {parent}"
    "AFTER/OVERLAP": ["{parent} happened after or overlap {child}"],
    # First one for event-event, second one for timex-event
    "CONTAINS": ["During {parent}, {child} happened"],
    "Depend-on": ["{child} depended on {parent}"],
    "BEGINS-ON": ["{parent} begins on {child}"],
    "ENDS-ON": ["{parent} ends on {child}"],
    # child contains parent
    "CONTAINS-INV": ["During {child}, {parent} happened"],
    # child is the test, parent is the result
    "NOTED-ON": ["The {child} test showed the result {parent}"],#, , "{parent} showed {child}."
}

TEMPLATE_TO_LABEL = {}
for k, v in TDG_LABEL_TEMPLATES.items():
    for t in v:
        TEMPLATE_TO_LABEL[t] = k

THYME_TEMPLATE_TO_LABEL = {}
for k, v in THYME_TDG_LABEL_TEMPLATES.items():
    for t in v:
        THYME_TEMPLATE_TO_LABEL[t] = k

# CHILD:PARENT
TDG_VALID_CONDITIONS = {
    "before": ["EVENT:TIMEX", "EVENT:EVENT", "EVENT:DCT", "TIMEX:TIMEX", "TIMEX:DCT"],#, "EVENT:ROOT"
    "after": ["EVENT:TIMEX", "EVENT:EVENT", "EVENT:DCT", "TIMEX:TIMEX", "TIMEX:DCT"],#, "EVENT:ROOT"
    "overlap": ["EVENT:EVENT", "EVENT:DCT"],#, "EVENT:ROOT"
    "included": ["EVENT:TIMEX", "TIMEX:TIMEX"],
    "Depend-on": ["TIMEX:TIMEX", "TIMEX:DCT", "EVENT:ROOT", "TIMEX:ROOT", "DCT:ROOT"]
}

# From parent to child
THYME_TDG_VALID_CONDITIONS = {
    "BEFORE": ["EVENT:TIMEX3", "TIMEX3:EVENT", "EVENT:EVENT", "DOCTIME:EVENT", "TIMEX3:TIMEX3", "DOCTIME:TIMEX3"],
    # , "EVENT:ROOT"
    "AFTER": ["EVENT:TIMEX3", "TIMEX3:EVENT", "EVENT:EVENT", "DOCTIME:EVENT", "TIMEX3:TIMEX3", "DOCTIME:TIMEX3"],
    # , "EVENT:ROOT"
    "OVERLAP": ["EVENT:EVENT", "EVENT:TIMEX3", "TIMEX3:EVENT", "DOCTIME:EVENT", "DOCTIME:TIMEX3"],  # , "EVENT:ROOT"
    "AFTER/OVERLAP": ["DOCTIME:EVENT"],

    "CONTAINS-SUBEVENT": ["EVENT:EVENT"],
    "CONTAINS-SUBEVENT-INV": ["EVENT:EVENT"],

    "NOTED-ON-INV": ["EVENT:EVENT"],
    "NOTED-ON": ["EVENT:EVENT"],

    "CONTAINS": ["TIMEX3:EVENT", "EVENT:EVENT", "TIMEX3:TIMEX3"],
    "CONTAINS-INV": ["EVENT:TIMEX3", "EVENT:EVENT", "TIMEX3:TIMEX3"],

    "BEGINS-ON": ["TIMEX3:EVENT", "EVENT:EVENT", "TIMEX3:TIMEX3", "EVENT:TIMEX3"],
    "ENDS-ON": ["TIMEX3:EVENT", "EVENT:EVENT", "TIMEX3:TIMEX3", "EVENT:TIMEX3"],

    "Depend-on": ["TIMEX3:TIMEX3", "DOCTIME:TIMEX3", "ROOT:EVENT", "ROOT:TIMEX3", "ROOT:DOCTIME"]
}

weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_abbre = ["Mon", "Tue", "Tues", "Wed", "Thu", "Thur", "Thurs", "Fri", "Sat", "Sun"]
month2num = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


def has_year(timex):
    return get_year(timex)


def get_year(timex_text):
    yr = re.search(r"[12]\d{3}", timex_text, flags=0)
    if yr:
        yr = yr.group()
    else:
        if "/" in timex_text:
            if len(timex_text.split("/")) == 3:
                m, d, y = timex_text.split("/")
                m = int(m)
                d = int(d)
                if 0 < m <= 12 and 0 < d <= 31 and len(y) == 2:
                    return y
        elif "." in timex_text:
            if len(timex_text.split(".")) == 3 and timex_text[0].isdigit():
                m1, d1, y1 = timex_text.split(".")
                m1 = int(m1)
                d1 = int(d1)
                if 0 < m1 <= 12 and 0 < d1 <= 31 and len(y1) == 2:
                    return y1
    return yr


def first_token_is_weekday(timex):
    first_token = timex.split()[0]
    return first_token in weekday or \
           first_token[:3] in weekday_abbre or \
           first_token[:4] in weekday_abbre or \
           first_token[:5] in weekday_abbre


def first_token_is_year(timex):
    first_token = timex.split()[0]
    w_compiled = re.search(r'\d+', first_token, flags=0)
    if w_compiled:
        year = w_compiled.group()
        if len(year) == 4 and int(year[0]) in [1, 2]:
            return True
        elif len(year) == 3 and int(year[0]) in [1, 2]:
            return True
    return False


def is_complete_month_date_year(timex):
    tokens = timex.split()
    if len(tokens) != 4:
        return False
    month, date, _, year = tokens
    return is_month(month) and is_day(date) and is_year(year)


def month_only(te):
    return te[:3].lower() in month2num and len(te.split()) == 1 and len(te.split("-")) == 1


def is_year(te):
    return te.isdigit() and len(te) == 4 and int(te[0]) in [1, 2]


def is_decade(te):
    return te.isdigit() and len(te) == 3 and int(te[0]) in [1, 2]


def is_month(te):
    return te[:3].lower() in month2num


def is_day(te):
    return te.isdigit() and len(te) in [1, 2] and int(te) in [i for i in range(1, 32)]
