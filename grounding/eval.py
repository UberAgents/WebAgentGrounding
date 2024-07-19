import os
import json
import logging
from datetime import datetime

from huggingface_hub import snapshot_download

import weblinx as wl
from weblinx.utils.recs import ungroup_dict_to_records

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))

def maybe_download(demo_names, dataset_dir):
    patterns = [f"demonstrations/{name}/*" for name in demo_names]
    # only download demonstration folders that do not exists
    patterns = [pattern for pattern in patterns if not os.path.exists(os.path.join(dataset_dir, pattern)[:-1])]

    snapshot_download(
        "McGill-NLP/WebLINX-full", repo_type="dataset", local_dir=dataset_dir, allow_patterns=patterns, ignore_patterns=["*.mp4"]
    )


def load_data(dataset_dir, demo_names):
    maybe_download(demo_names, dataset_dir)

    demo_dir = os.path.join(dataset_dir, "demonstrations")
    demos = [wl.Demonstration(demo_name, base_dir=demo_dir) for demo_name in demo_names]
    return demos


def recall_at_k(input_records, k, label_key="label", rank_key="rank"):
    num_correct = 0
    num_total = 0

    for r in input_records:
        if r[label_key] == 1:
            num_total += 1
            if r[rank_key] <= k:
                num_correct += 1

    score = num_correct / num_total
    return score


def mean_reciprocal_rank(input_records, label_key="label", rank_key="rank", k=None):
    if k is None or len(input_records) < k or k < 1:
        k = len(input_records)

    mrr = 0
    num_total = 0

    for r in input_records:
        if r[label_key] == 1:
            if r[rank_key] <= k:
                mrr += 1 / r[rank_key]
            num_total += 1

    mrr /= num_total

    return mrr


def calculate_ranking_metrics(scores, demos, split):
    results = {
        "split": split,
        "num_turns": len(scores),
        "num_demos": len(demos),
        "mrr": mean_reciprocal_rank(scores, k=50),
    }

    ks = [1, 5, 10, 20, 50, 100, 200]
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(scores, k=k)

    return results


def evaluate(grounding_model, split, model_name=""):
    dataset_dir = os.path.join(BASE_DIR, "datasets/wl_data/")

    # load data
    if split == "testing":
        demo_names = [
            "ygprzve",
            "saabwsg",
            "iqaazif",
            "eiblold",
            "pjzqiar",
            "polclhz",
            "hvdwmnq",
            "feupcgi",
            "hdbpxqn",
            "bdhiwrz",
            "bonfxww",
            "vyetbcl",
            "eaozdtr",
            "dpftfrs",
            "qjmnlfs",
            "aoxxcdg",
            "orlmkas",
        ]
    else:
        split_path = os.path.join(dataset_dir, "splits.json")
        demo_names = wl.utils.load_demo_names_in_split(split_path, split=split)

    demos = load_data(dataset_dir, demo_names)

    # process data with own model
    candidates = grounding_model.predict(demos, group_scores = True)
    scores = ungroup_dict_to_records(candidates)

    # calculate metrics
    results = calculate_ranking_metrics(scores, demos, split)

    logging.info(results)

    # Save results
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    result_dir = f"results/{model_name}_" + now
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {result_dir}")
