import abc
import logging
from tqdm import tqdm
from typing import List

from weblinx.processing import group_record_to_dict

from grounding.preprocessing import build_records_for_single_demo, build_formatters


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


class GroundingModel(abc.ABC):
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

    @abc.abstractmethod
    def predict_score(self, samples):
        # return list of scores
        pass

    def predict(self, demos, group_scores=False):
        samples = self.preprocess(demos)
        preds = self.predict_score(samples, group_scores=group_scores)
        return preds