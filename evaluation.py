import pandas as pd
from typing import List, Dict, Tuple
import json


def intersect_two_ranges(range1: Tuple[int, int], range2: Tuple[int, int]) -> Tuple[int, int] | None:
    start1, end1 = range1
    start2, end2 = range2
    start = max(start1, start2)
    end = min(end1, end2)
    return (start, end) if start < end else None


def sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def union_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def difference(ranges: List[Tuple[int, int]], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    result = []
    target_start, target_end = target
    for start, end in ranges:
        if end <= target_start or start >= target_end:
            result.append((start, end))
        elif start < target_start and end > target_end:
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            result.append((start, target_start))
        elif end > target_end:
            result.append((target_end, end))
        # Else: the entire chunk is covered, remove it
    return result


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using start-end spans and overlap-based precision/recall.
    """

    def __init__(self, questions_df: pd.DataFrame, document: str):
        self.questions_df = questions_df.copy()
        self.document = document

    def extract_references(self, references: str | List[Dict]) -> List[Dict]:
        if isinstance(references, str):
            try:
                return json.loads(references)
            except Exception:
                return []
        return references

    def evaluate(self, query: str, retrieved_chunks: List[str], corpus_id: str) -> Dict[str, float]:
        # Filter correct question row
        question_row = self.questions_df[
            (self.questions_df["question"] == query) &
            (self.questions_df["corpus_id"] == corpus_id)
        ]

        if question_row.empty:
            return {"precision": 0.0, "recall": 0.0}

        row = question_row.iloc[0]
        references = self.extract_references(row["references"])

        golden_spans = [(int(ref["start_index"]), int(ref["end_index"])) for ref in references]
        unused_highlights = golden_spans.copy()

        matched_spans = []
        chunk_ranges = []

        for chunk in retrieved_chunks:
            start_index = self.document.find(chunk)
            if start_index == -1:
                continue
            end_index = start_index + len(chunk)
            chunk_range = (start_index, end_index)
            chunk_ranges.append(chunk_range)

        # Evaluate overlap
        for chunk_range in chunk_ranges:
            for ref_range in golden_spans:
                intersection = intersect_two_ranges(chunk_range, ref_range)
                if intersection:
                    matched_spans = union_ranges([intersection] + matched_spans)
                    unused_highlights = difference(unused_highlights, intersection)
                    break  # Move to next chunk

        # Precision
        if chunk_ranges:
            precision_numerator = sum_of_ranges(matched_spans)
            precision_denominator = sum_of_ranges(chunk_ranges)
            precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0.0
        else:
            precision = 0.0

        # Recall
        recall_numerator = sum_of_ranges(matched_spans)
        recall_denominator = sum_of_ranges(golden_spans)
        recall = recall_numerator / recall_denominator if recall_denominator > 0 else 0.0

        return {"precision": precision, "recall": recall}
