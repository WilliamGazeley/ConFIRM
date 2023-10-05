# Contains sample questions for some given data/field, used to seed the 
# question generator.

# Seeding template uses format:
# Tuple[<information>, <question>]

# Replace the following with your own samples

MIXED = [
    (
        "company_name\n"
        "field_name - field description",
        "Question: A good question sample",
    ),
    (
        "company_name_2\n"
        "field_name_2 - field description",
        "Question: A good question sample",
    ),
    # Add more samples here
]

EXTERNAL_DATA = [
    # This is to add some question examples for external fields only
    (
        "external field name - external field description",
        "a good question sample",
    ),
    (
        "external field name - external field description",
        "a good question sample",
    ),
    (
        "external field name - external field description",
        "a good question sample",
    ),
]


ALL_SEEDS = MIXED + EXTERNAL_DATA

if __name__ == "__main__":
    # Run this file to validate the samples
    for sample in ALL_SEEDS:
        assert len(sample) == 2, f"Invalid sample: {sample}"
    print(f"Successfully validated {len(ALL_SEEDS)} samples")
