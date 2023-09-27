import re
import pandas as pd
from typing import List, Callable
from functools import partial


def llm_invalid(
        df: pd.DataFrame, columns: List[str],
        axis: int = 0) -> pd.DataFrame:
    """
    Filter out generated questions that cannot be handled by an LLM.
    """
    assert axis in [0, 1], "Axis must be 0 or 1"
    
    blacklist = ["image", "images", "graph", "graphs", "picture", "pictures",
                 "file", "files", "map", "maps", "draw", "plot", "go to"]

    def find_word_in_string(w, s):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)

    for column in columns:
        mask = df[column].apply(lambda question: not any(
            find_word_in_string(word, question) for word in blacklist))
        df = df[mask]

    return df
