# Create a set of questions based on some information and seeded using some
# examples.

from langchain.chat_models import ChatOpenAI
from lora_confirm import question_generator, filters

def test_seeded():
    n = 5  # Number of questions to generate
    llm = ChatOpenAI()
    fields = ["animal.cat - contains trivia about cats",
            "animal.dog - contains trivia about dogs",
            "cities.london - contains trivia about London"]

    prompt_samples = [
        (fields[0], "What civilization thought cats were divine?"),
        (fields[1], "What is the most popular breed of dog?"),
        (fields[2], "What is the total population of London?"),
    ]

    qs = question_generator.seeded(llm=llm, fields=fields, n=n, k_samples=2,
                                prompt_samples=prompt_samples)
    assert len(qs) == n, f"Expected {n} questions, got {len(qs)}"
    print(f"Generated:\n{qs}")
    
    qs = filters.rouge_l(qs, columns=['question'])
    assert len(qs) < n, f"Expected {n} questions, got {len(qs)}"
    print(f"Filtered:\n{qs}")
