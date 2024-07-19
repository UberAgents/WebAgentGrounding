# clone of https://github.com/McGill-NLP/weblinx/blob/main/modeling/dmr/processing.py

from collections import defaultdict
from copy import deepcopy
import random
from typing import Any, Dict, List
from functools import partial

import lxml.html
import weblinx as wl
import weblinx.utils.html as wh
import weblinx.utils.format as wlf
from weblinx.processing.prompt import (
    format_prev_turns,
    find_turns_with_instructor_chat,
    format_utterances,
)
from bs4 import BeautifulSoup

from grounding.preprocessing import turn_has_valid_uid, format_turn_for_input, represent_element_as_dict, convert_elem_dict_to_str_legacy


def get_elements_with_n_childs(html_content, n):
    soup = BeautifulSoup(html_content, 'html.parser')
    small_elements = []
    for e in soup.find_all():
        childs = [c for c in e.children]
        l = len(childs)
        if l > n:
            continue

        if l == 1:
            try:
                l_2 = len([c for c in childs[0].children])
            except:
                l_2 = 0
            if l_2 > 0:
                continue

        small_elements.append(e)

    small_keys = [e.get("data-webtasks-id") for e in small_elements]
    return small_keys


def get_elements_wo_parents(html_content, keys):
    soup = BeautifulSoup(html_content, 'html.parser')
    # keys = [e.get("data-webtasks-id") for e in soup.find_all()]
    e_txts = [soup.find(attrs={"data-webtasks-id": k}) for k in keys]
    child_keys = []
    for k, txt in zip(keys, e_txts):
        # only append if there is no other txt that contains this text
        if txt is None:
            continue
        
        has_child_txt = False
        for o_k, o_txt in zip(keys, e_txts):
            if k == o_k or o_txt is None or len(o_txt) < 1:
                continue
            if str(o_txt) in str(txt) and not o_txt == txt:
                # print(k, o_k)
                # print(o_txt[:10], txt[:10])
                has_child_txt = True
                break

        if not has_child_txt:
            child_keys.append(k)

    return child_keys


def build_records_for_single_turn(
    turn, replay, format_intent_input, uid_key, max_neg=None, only_allow_valid_uid=True
) -> List[dict]:
    """
    This function will build a list of dictionaries, each of which is a record
    for a single turn. Each record has the following keys:
        - query: the dialogue history, used as a query for training the model
        - doc: concise representation of HTML element used as doc for training
        - label: either 0 or 1, indicating whether the document is the target element
        - uid: the unique identifier for an element, must be in the element attributes
        - turn_index: the index of the turn in the replay
        - demo_name: the name of the demonstration

    If `only_allow_valid_uid` is True, then only turns that have a valid uid
    will be included in the output. Otherwise, all turns will be included.
    """
    bboxes_filt = wh.filter_bboxes(
        turn.bboxes,
        viewport_height=turn.viewport_height,
        viewport_width=turn.viewport_width,
    )
    root = lxml.html.fromstring(turn.html)
    root_tree = root.getroottree()
    elements = root.xpath(f"//*[@{uid_key}]")
    elements_filt = [p for p in elements if p.attrib[uid_key] in bboxes_filt]

    small_keys = get_elements_with_n_childs(turn.html, 1)
    # small_keys = get_elements_wo_parents(turn.html, bboxes_filt)
    elements_filt = [e for e in elements_filt if e.attrib[uid_key] in small_keys]


    has_valid_uid = turn_has_valid_uid(turn, paths=elements, uid_key=uid_key)
    if only_allow_valid_uid and not has_valid_uid:
        return []

    # Now, we can format each of the elements in paths_filt into string
    # and use them as negative samples
    query = format_turn_for_input(replay, turn, format_intent=format_intent_input)
    target_uid = turn.element["attributes"][uid_key] if has_valid_uid else -1

    records_positive = []
    records_negative = []

    for elem in elements_filt:
        bbox = turn.bboxes[elem.attrib[uid_key]]
        elem_dict = represent_element_as_dict(elem, bbox, root_tree)
        elem_str = convert_elem_dict_to_str_legacy(elem_dict)

        record = {
            "query": query,
            "doc": elem_str,
            "uid": elem.attrib[uid_key],
            "demo_name": turn.demo_name,
            "turn_index": turn.index,
            "elem_dict": elem_dict,
        }

        if elem.attrib[uid_key] == target_uid:
            record["label"] = 1
            records_positive.append(record)
        else:
            record["label"] = 0
            records_negative.append(record)

    if max_neg is not None and 0 < max_neg < len(records_negative):
        records_negative = random.sample(records_negative, max_neg)

    return records_positive + records_negative


def build_records_for_single_demo(
    demo,
    format_intent_input,
    max_neg_per_turn=None,
    random_state=None,
    uid_key="data-webtasks-id",
    only_allow_valid_uid=True,
    group_by_turn=False,
) -> List[dict]:
    """
    This runs `build_records_for_single_turn` for each turn in the demonstration.
    First, the demonstration is converted into a replay, and then we filter the
    turns to only those that have HTML and bounding boxes, and that are of the
    following intents:
        - click
        - change
        - textInput
        - scroll
        - load
        - submit

    Any turn that does not have a valid uid is discarded.
    """
    if random_state is not None:
        random.seed(random_state)

    replay = wl.Replay.from_demonstration(demo)
    turns = replay.filter_by_intents(
        "click", "change", "textInput", "scroll", "load", "submit"
    )
    turns = wl.filter_turns(turns, lambda t: t.has_html() and t.has_bboxes())

    records_for_demo = []
    for turn in turns:
        recs = build_records_for_single_turn(
            turn=turn,
            replay=replay,
            format_intent_input=format_intent_input,
            uid_key=uid_key,
            max_neg=max_neg_per_turn,
            only_allow_valid_uid=only_allow_valid_uid,
        )
        if group_by_turn:
            records_for_demo.append(recs)
        else:
            records_for_demo.extend(recs)

    return records_for_demo
