from lora_confirm import filters
import pandas as pd


def test_rouge_l():
    df = pd.DataFrame({"A": ["What is the capital of France?",
                             "What is the capital of France?",
                             "What is the capital of Germany?",
                             "What is the capital of Germany?",
                             "What is the capital of Spain?",
                             "What is the capital of France?"],
                       "B": ["Foo",
                             "Foo Foo",
                             "Bar",
                             "The capital of Germany is Berlin",
                             "Bar",
                             "What is the capital of France?"]})
    # Test single verticle
    expected_output = pd.DataFrame({"A": ["What is the capital of France?"],
                                    "B": ["Foo"]})
    result = filters.rouge_l(df, ['A'], threshold=0.7).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output)

    # Test multiple verticle
    expected_output = pd.DataFrame({"A": ["What is the capital of France?",
                                          "What is the capital of Germany?"],
                                    "B": ["Foo",
                                          "Bar"]})
    result = filters.rouge_l(
        df, ['A', 'B'],
        threshold=0.9).reset_index(
        drop=True)
    pd.testing.assert_frame_equal(result, expected_output)

    # Test single-pair horizontal
    expected_output = pd.DataFrame({"A": ["What is the capital of France?",
                                          "What is the capital of France?",
                                          "What is the capital of Germany?",
                                          "What is the capital of Spain?"],
                                    "B": ["Foo",
                                          "Foo Foo",
                                          "Bar",
                                          "Bar"]})
    result = filters.rouge_l(
        df, ['A', 'B'],
        axis=1, threshold=0.3).reset_index(
        drop=True)
    pd.testing.assert_frame_equal(result, expected_output)

    # Test multiple horizontal
    df['C'] = ["Buzz", "Buzz", "Buzz", "Buzz", "Bar", "Buzz"]
    expected_output = pd.DataFrame({"A": ["What is the capital of France?",
                                          "What is the capital of France?",
                                          "What is the capital of Spain?"],
                                    "B": ["Foo",
                                          "Foo Foo",
                                          "Bar"],
                                    "C": ["Buzz",
                                          "Buzz",
                                          "Buzz", ]})


def test_duplicates():
    # Test axis=0
    df = pd.DataFrame({"A": ["What is the capital of France?",
                             "What is the capital of Germany?",
                             "What is the capital of Germany?"],
                       "B": ["Foo", "Bar", "Buzz"]})
    expected_output = pd.DataFrame({"A": ["What is the capital of France?",
                                          "What is the capital of Germany?"],
                                    "B": ["Foo", "Bar"]})
    result = filters.duplicates(df, ['A']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output)

    # Test axis=1
    expected_output = df
    result = filters.duplicates(df, ['A', 'B'], axis=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output)


def test_llm_invalid():
    df = pd.DataFrame({"A": ["Foo", "Bar", "image"],
                       "B": ["Foo", "map", "Buzz"]})
    expected_output = pd.DataFrame({"A": ["Foo"],
                                    "B": ["Foo"]})

    # This is a blacklist filter, so axis=0 and axis=1 behave the same
    result = filters.llm_invalid(df, ['A', 'B']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output)

    result = filters.llm_invalid(df, ['A', 'B'], axis=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output)
