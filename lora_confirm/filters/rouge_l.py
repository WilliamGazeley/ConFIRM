import re
import pandas as pd
from typing import List, Callable
from itertools import combinations


def _rouge_l_axis_0(
        df: pd.DataFrame, columns: List[str],
        threshold: float) -> pd.DataFrame:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    drop_indices = []

    for column in columns:
            
        questions = df[column].tolist()
        filtered_questions = []
        for i, question in enumerate(questions):
            if not any(
                    scorer.score(existing_question, question)['rougeL'].fmeasure >
                    threshold for existing_question in filtered_questions):
                filtered_questions.append(question)
            else:
                drop_indices.append(i)
    
    if threshold < 1: # Edge case, which allows for identical questions
        for column in columns:
            df = df.drop_duplicates(subset=[column], keep='first')

    df = df.drop(drop_indices, errors='ignore')
    return df


def _rouge_l_axis_1(
        df: pd.DataFrame, columns: List[str],
        threshold: float) -> pd.DataFrame:
    from rouge_score import rouge_scorer

    assert len(columns) >= 2, "Must provide at least two columns"

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def is_below_threshold(row):
        for column_a, column_b in combinations(columns, 2):
            score = scorer.score(row[column_a], row[column_b])[
                'rougeL'].fmeasure
            if score > threshold:
                return False
        return True

    mask = df.apply(is_below_threshold, axis=1)
    return df[mask]


def rouge_l(
        df: pd.DataFrame, columns: List[str],
        axis: int = 0, threshold: float = 0.7) -> pd.DataFrame:
    """
    Filter out generated questions that are word-for-word too similar to the
    other questions.

    Args:
        df: DataFrame containing the questions to filter
        columns: Name of the columns containing the columns to filter by
        axis: Axis to apply the filter on. 0 for vertical, 1 for horizontal.
        threshold: Rouge-L threshold to use for filtering. 0.7 is the threshold
            used by the self-instruct paper.
            *Note: threshold=1 means questions can be identical.
    """
    if axis == 0:
        return _rouge_l_axis_0(df, columns, threshold)
    elif axis == 1:
        return _rouge_l_axis_1(df, columns, threshold)
    else:
        raise ValueError("Axis must be 0 or 1")
