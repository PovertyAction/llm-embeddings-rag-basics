# Session 02 — Embeddings & RAG

## Guided Activity

### Preparation
Make sure you have:

- active environment (`.venv`),
- `.env` with `OPENAI_API_KEY`,
- the repo open in VS Code.


### Step A — Generate embeddings (10 min)
Run:
```bash
python examples/01_generate_embeddings.py
```

What you should see:
- print of the vector size,
- preview of the first values,
- confirmation that the API works for embeddings.


### Step B — Simple semantic search (15 min)
Run:
```bash
python examples/02_similarity_search.py
```

This script:
1) reads several short documents,
2) splits them into chunks (simple chunking),
3) generates embeddings per chunk,
4) generates embedding of a question,
5) calculates similarities and shows the top-k.

What to observe:
- the retrieved chunks should "make sense" with your question,
- the top-1 and top-3 should be reasonable.


### Step C — Mini-RAG (10–15 min)
Run:
```bash
python examples/03_mini_rag.py
```

This script does the same as Step B, but additionally:
- builds a prompt that includes the retrieved chunks,
- asks the LLM to respond using **only that context**,
- prints the response.

What to observe:
- the response should reference the retrieved content,
- if the question is not covered by the context, the model should say so (according to the instruction).

---

## 3) Quick exercises (if time allows)

1) Change `TOP_K` from 3 to 5 and compare results.
2) Change the chunking (e.g. by paragraphs vs. fixed size) and see how the top-k changes.
3) Write a "difficult" question (little evidence in the docs) and observe how the mini-RAG behaves.

---

## 4) Bridge to the next sessions

What comes next is usually:
- more robust chunking (tokens, overlap),
- storing embeddings (files / vector DB),
- traceability (citations, chunk IDs),
- evaluation: are we retrieving the right thing? does the response use the context?

This repository already leaves you with the technical core to move forward.
