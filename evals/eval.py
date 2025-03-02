from eval_utils import MultiHopEval

CHARS_PER_TOKEN = 3
datapoint = MultiHopEval.make_one(
    n_chars_problem=int(1_000_000 * CHARS_PER_TOKEN),
    num_queries=5,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)
print(datapoint.prompt)
print(datapoint.completion)
print(datapoint.targets)
