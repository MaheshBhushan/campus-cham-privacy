# Local LLM rewrite of indirect identifiers (tested alternative)

Question: can a small language model, running locally so no ticket leaves the machine, rewrite self-descriptions
("I am a 26-year-old female student from Iran in MMC") better than the rule-based step in `ccp.indirect`?

Setup: `ccp.indirect.llm_rewrite`, Qwen2.5 Instruct via Hugging Face transformers on CPU, greedy decoding
(reproducible), two few-shot examples, the system prompt in `src/ccp/indirect.py`. Input: Presidio output of the
first 16 tickets with a self-description (seed 42). Reproduce with `uv run ccp llm-rewrite --limit 16`.

| Model | Time per text (CPU, machine under load) | Usable? |
|---|---|---|
| Qwen2.5-0.5B-Instruct | ~11 s | No: kept age and programme, switched English texts to German, or echoed the input |
| Qwen2.5-1.5B-Instruct | ~62 s (≈ 2 h for all 113 texts) | Partly |

Result on the 16 texts (1.5B):

| | Writer unique | Median anonymity set |
|---|---|---|
| Presidio output | 0 % | 48 |
| Rule-based step | 0 % | 567 |
| LLM rewrite | 0 % | 454 |

Manual review of the 16 LLM outputs:

- 6 invented a gender ("Als Studentin …" implies a woman wrote it), i.e. it added a quasi-identifier;
- 2 kept the exact age and programme; 1 kept "MDM", 1 kept "male";
- 1 dropped the actual request;
- 4 changed the meaning ("German universities" → "other universities");
- about 3 of 16 were clean rewrites.

Conclusion for the journal: on a laptop CPU the small local model is slower than the rules by four orders of
magnitude, less private on the anonymity-set metric, and changes the message in a quarter of cases. A larger model
on a GPU server inside the university might do better, but would need the same evaluation before use; the
anonymity-set metric also only sees what its lexicons recognise, which flatters free-form LLM wording.
