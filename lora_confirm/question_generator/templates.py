GENERATION_PROMPT = (
    "You are a layman who knows little about finance. "
    "Generate a question that can be answered by and only by the following "
    "information. You MUST NOT include names of the available "
    "Information:\n{info}\n\n"
    "Question: "
)

SEEDING_GENERATION_PROMPT = (
    "You are a everyday investor who is member of the general public not a "
    "professional investor. Generate a question that can be answered by and "
    "only by the following information source or database field. Use "
    "informal conversational tone. You MUST NOT include names of the "
    "available.\n\n"
    "{examples}\n"
    "Information:\n{info}\n"
    "Question: "
)

SEEDING_EXAMPLE_PAIR = (
    "Information:\n{info}\n"
    "Question: {question}\n"
)
