# code largely taken from https://github.com/McGill-NLP/weblinx/tree/main/modeling/dmr

import logging
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score
from weblinx.utils.recs import ungroup_dict_to_records, group_record_to_dict

from grounding.preprocessing import build_formatters
from grounding.eval import evaluate
from grounding.model import GroundingModel

from preprocesing import build_records_for_single_demo


def run_model_and_update_groups(
    model, input_grouped: Dict[Any, List[dict]], batch_size, sim_method="cos_sim"
):
    if sim_method == "cos_sim":
        sim_func = cos_sim
    elif sim_method == "dot_product":
        sim_func = dot_score
    else:
        raise ValueError(f"Unknown similarity function: {sim_method}")

    for k, group in tqdm(input_grouped.items(), desc="Computing scores"):
        group = input_grouped[k]
        query = group[0]["query"]
        docs = [r["doc"] for r in group]

        encoded = model.encode(
            [query] + docs, batch_size=batch_size, show_progress_bar=False
        )
        query_vector, doc_vectors = encoded[0], encoded[1:]
        scores = sim_func(query_vector, doc_vectors).cpu().squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]

        for i, r in enumerate(group):
            r["score"] = scores[i]


def get_ranks_from_scores(scores: Dict[Any, float], starts_at=1) -> Dict[Any, int]:
    """
    Given a dictionary of key -> scores, return a dictionary of key -> ranks.
    """
    # Get sorted keys
    keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    ranks = {k: i + starts_at for i, k in enumerate(keys)}

    return ranks


def verify_queries_are_all_the_same(grouped_records: dict) -> bool:
    """
    Given a dictionary of grouped records, this function verifies that all
    queries are the same within each group.
    """
    for k, v in grouped_records.items():
        first_query = v[0]["query"]
        if not all(r["query"] == first_query for r in v):
            return False
    return True


class WeblinxGrounding(GroundingModel):
    def preprocess(self, demos):
        format_intent_input, _ = build_formatters()
        input_records: List[dict] = []
        logging.info(f"Number of demos: {len(demos)}. Starting building records.")
        for demo in tqdm(demos, desc="Building input records"):
            demo_records = build_records_for_single_demo(
                demo=demo,
                format_intent_input=format_intent_input,
                max_neg_per_turn=None,
                only_allow_valid_uid=True,
            )
            input_records.extend(demo_records)
        logging.info(f"Completed. Number of input records: {len(input_records)}")

        # Group records by (demo_name, turn_index) pairs
        input_grouped = group_record_to_dict(
            input_records, keys=["demo_name", "turn_index"], remove_keys=False
        )
        # Verify that queries are all the same within each group
        error_msg = "Queries are not all the same within each group"
        assert verify_queries_are_all_the_same(input_grouped), error_msg

        return input_grouped

    def predict_score(self, samples: dict, group_scores: bool = False):
        huggingface_model = "McGill-NLP/bge-small-dmr" # currently highest on leaderboard
        model = SentenceTransformer(huggingface_model)

        # default settings
        sim_method = "cos_sim"
        torch_dtype = torch.bfloat16
        use_amp = False
        batch_size = 64

        logging.info(f"Using the following similarity method: {sim_method}")

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch_dtype):
            run_model_and_update_groups(
                model, input_grouped=samples, batch_size=batch_size, sim_method=sim_method
            )
        logging.info("Completed")

        for group in samples.values():
            scores = {r["uid"]: r["score"] for r in group}
            ranks = get_ranks_from_scores(scores)
            for r in group:
                r["rank"] = ranks[r["uid"]]

        if group_scores:
            input_records = samples
        else:
            input_records = ungroup_dict_to_records(samples)

        return input_records


if __name__ == "__main__":
    model = WeblinxGrounding()
    evaluate(model, split="testing", model_name="webtree")