import time
import pandas as pd
from langchain.llms import BaseLLM
from langchain.schema import SystemMessage
from typing import Iterable, Callable, Any, List, Tuple, Dict, Awaitable
from random import choice, sample
import asyncio
import logging

from .templates import (GENERATION_PROMPT, SEEDING_GENERATION_PROMPT,
                        SEEDING_EXAMPLE_PAIR)

log = logging.getLogger(__name__)


def simple(
    llm: BaseLLM,
    fields: Iterable[str],
    n: int = 2,
    max_combo: int = 2,
    prompt: str = GENERATION_PROMPT,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: question, expected_fields

    Args:
        llm: Model to use for question generation
        fields: List of all fields allowed to be used in the question
        n: Number of questions to generate
        max_combo: Maximum number of fields to use when generating one question
        prompt: System prompt for question generation
    """
    max_combo = min(max_combo, len(fields))
    combos = [x+1 for x in range(max_combo)]
    selected_fields = [sample(fields, k=choice(combos)) for _ in range(n)]
    prompts = [prompt.format(info=x) for x in selected_fields]
    resp = llm.batch(prompts)  # I don't know how Langchain error handles batch
    questions = [x.content for x in resp]
    columns = ['question', 'expected_fields']
    df = pd.DataFrame(data=[x for x in zip(questions, selected_fields)],
                      columns=columns)
    return df


def seeded(
    llm: BaseLLM,
    fields: Iterable[str],
    prompt_samples: List[Tuple[str, str]],
    subjects: Iterable[str] = [],
    n: int = 2,
    k_samples: int = 3,
    max_combo: int = 2,
    max_retries: int = 2,
    prompt: str = SEEDING_GENERATION_PROMPT,
    save_path: str = None
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: [question, expected_fields], using a 
    random sample of example questions (few-shot). Optionally, subjects can be
    provided which will generate questions based on fields with regard one 
    subject.

    Ref: https://arxiv.org/abs/2212.10560

    Args:
        llm: Model to use for question generation
        fields: List of all fields allowed to be used in the question
        prompt_samples: List of example info-question pairs to seed the model
        subjects: List of subjects allowed to be used in the question
        n: Number of questions to generate
        k_samples: Number of examples to use for seeding
        max_combo: Maximum number of fields to use when generating one question
        max_retries: Maximum number of retries if generation fails
        prompt: System prompt for question generation
        save_path: Path to save the generated questions, if None, don't save

    Raises:
        ValueError: if save_path is not None and does not end with '.csv'.
        Exception: if question generation fails after max_retries attempts.
    """
    assert all([len(x) == 2 for x in prompt_samples]), \
        "prompt_samples must be a list of tuples of length 2"
    assert k_samples <= len(prompt_samples), f"Seeding samples should be larger than {k_samples}. Add more examples in seeds file."
    if save_path and not save_path.endswith('.csv'):
        raise ValueError("save_path must be a .csv")
    # To avoid branching the prompt building code below, modify the subjects
    subjects = [''] if not subjects else [x + "\n" for x in subjects]

    max_combo = min(max_combo, len(fields))

    for attempt in range(max_retries):
        try:
            selected_fields = [
                sample(fields, k=choice(range(1, max_combo+1)))
                for _ in range(n)
            ]

            prompts = [
                prompt.format(
                    info=choice(subjects) + "\n".join(field_set),
                    examples="\n".join([
                        SEEDING_EXAMPLE_PAIR.format(info=ex_info, question=ex_question)
                        for ex_info, ex_question in sample(prompt_samples, k=k_samples)
                    ])
                )
                for field_set in selected_fields
            ]

            responses = llm.batch(prompts, temperature=0.5)
            questions = [response.content.strip() for response in responses]
            # weaker models repeat the prompt, so parse it out
            questions = [x.replace("Question: ", "") for x in questions]

            df = pd.DataFrame(
                data=list(zip(prompts, questions, selected_fields)),
                columns=['prompts', 'question', 'expected_fields']
            )

            if save_path:
                df.to_csv(save_path)

            return df

        except Exception as e:
            log.error("An error occurred", exc_info=True)
            print(f"Generation failed on attempt {attempt+1}, retrying...")
            time.sleep(5)  # TODO: Consider adding a backoff

    raise Exception("Question generation failed after maximum retries.")
