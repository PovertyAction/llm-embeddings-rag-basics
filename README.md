# Session 02 – Introduction to Embeddings & RAG Models

## Session goal

By the end of this session, you will be able to:

- understand what embeddings are and why they matter,
- understand what a RAG (Retrieval Augmented Generation) model is,
- recognize when to use embeddings vs. simple LLM calls,
- understand the architecture of knowledge-augmented applications.

This session focuses on **conceptual foundations for working with your own data**.

---

## What we are (and are not) doing today

### We are doing

- Understanding embeddings conceptually
- Learning the RAG architecture pattern
- Seeing practical examples of when these tools are useful
- Understanding the workflow from text to searchable knowledge

### We are not doing

- Building applications (that's for later sessions)
- Writing code for qualitative coding or chatbots
- Working with vector databases
- Fine-tuning models

Think of this session as **understanding the theory before building**.

---

## Prerequisites

Before starting this session, you should have completed Session 01:

- ✅ Local Python environment configured
- ✅ API key set up and working
- ✅ Successfully run `01_test_connection.py`

---

## Big picture: from simple LLM to RAG

In Session 01, we learned:
```
User prompt → LLM → Response
```

In this session we understand a more powerful pattern:
```
User question 
↓
Search relevant documents (using embeddings)
↓
Build context-aware prompt
↓
LLM → Response based on YOUR data
```

This is the RAG pattern.

---

## Part 1: What are embeddings?

### The problem
LLMs can't naturally "search" through thousands of documents.
They need text converted into a format that allows mathematical comparison.

### The solution: embeddings
An **embedding** is a numerical representation of text.

- Text → Vector of numbers (e.g., [0.2, -0.5, 0.8, ...])
- Similar meanings → Similar vectors
- Different meanings → Distant vectors

### Why this matters
With embeddings, you can:

- Find documents similar to a query
- Cluster related texts together
- Measure semantic similarity mathematically
- Search by meaning, not just keywords

### Example: How meaning becomes numbers

```
Text: "The survey was conducted in rural villages"
↓
Embedding: [0.023, -0.145, 0.892, ..., 0.334]
(1536 numbers total)
```
```
Similar text: "Data collection happened in countryside communities"
↓
Embedding: [0.019, -0.151, 0.887, ..., 0.329]
(very similar numbers!)
```
```
Different text: "The stock market crashed yesterday"
↓
Embedding: [0.876, 0.234, -0.445, ..., -0.123]
(very different numbers)
```

### Key insight
**Embeddings capture semantic meaning, not just words.**

- "car" and "automobile" have similar embeddings
- "bank" (financial) and "bank" (river) have different embeddings
- Works across languages in multilingual models

---

## Part 2: What is a RAG model?

**RAG = Retrieval Augmented Generation**

### The fundamental problem RAG solves
LLMs have two limitations:

1. **Knowledge cutoff**: They don't know about events after training
2. **No access to private data**: They don't know YOUR documents, reports, or internal knowledge

RAG fixes both by connecting LLMs to external knowledge sources.

### The three steps of RAG

#### 1. Retrieval
Search your documents for relevant content

- Convert user question to embedding
- Find documents with similar embeddings
- Retrieve top-k most relevant passages

#### 2. Augmentation
Add retrieved content to the prompt

- Build a context-aware prompt
- Include relevant documents as background
- Guide the LLM with specific information

#### 3. Generation
LLM generates response

- Model has access to YOUR data
- Answers are grounded in retrieved documents
- Can cite specific sources

### Visual representation

```
Your documents → Embeddings → Vector database
                                    ↓
User question → Embedding → Search similar → Top matches
                                                  ↓
                                    Build prompt with context
                                                  ↓
                                            LLM generates
                                                  ↓
                                          Grounded answer
```

---

## When to use embeddings vs. simple LLM calls

### Use simple LLM calls when:

- Task is general (translation, summarization, writing)
- No specific knowledge base needed
- Instructions fit in the prompt
- One-off tasks

### Use embeddings + RAG when:

- Working with large document collections
- Need to reference specific internal knowledge
- Information changes frequently
- Want to cite sources
- Building knowledge base applications

### Research use cases for RAG

- **Qualitative analysis**: Search similar interview responses
- **Literature review**: Find relevant papers from your database
- **Project knowledge**: Answer questions about past projects
- **Policy analysis**: Search regulations and guidelines
- **Data documentation**: Find relevant codebooks and data dictionaries

---

## Key concepts to understand

### 1. Embedding models

- Separate from LLMs
- Specifically trained to create good vector representations
- Example: `text-embedding-3-small` from OpenAI
- Different models for different languages/domains

### 2. Vector similarity

- **Cosine similarity**: measures angle between vectors
- Score from -1 to 1 (higher = more similar)
- Used to rank document relevance
- 0.8+ typically means very related content

### 3. Chunking strategy
Documents are split into smaller pieces:

- Each chunk gets embedded separately
- Balances between context and precision
- Typical size: 200-1000 words per chunk
- Must decide: semantic chunks vs. fixed-size chunks

### 4. Retrieval quality
The success of RAG depends on finding the RIGHT documents:

- Too few documents → miss relevant info
- Too many documents → confuse the LLM
- Sweet spot: typically 3-5 most relevant chunks
- Quality of retrieval > quantity of documents

---

## RAG vs. other approaches

### RAG vs. Fine-tuning
**Fine-tuning**: Retrain the model on your data

- Pros: Model "learns" your domain
- Cons: Expensive, time-consuming, hard to update

**RAG**: Search your data at query time

- Pros: Easy to update, works with any LLM, transparent
- Cons: Depends on retrieval quality

### RAG vs. Long context windows

**Long context**: Put everything in the prompt

- Pros: Simple, no separate retrieval step
- Cons: Expensive, slower, may miss details in long contexts

**RAG**: Only retrieve relevant parts

- Pros: Efficient, scales to large datasets
- Cons: Requires embedding infrastructure

### When to use each

- Small, static knowledge base → Long context
- Large, changing knowledge base → RAG
- Need model to "speak" your domain → Fine-tuning
- Most research use cases → RAG

---

## The RAG workflow in practice

### One-time setup

```
1. Collect your documents
2. Split into chunks
3. Create embeddings for each chunk
4. Store in searchable format
   (can be as simple as a CSV with embeddings)
```

### For each user question

```
1. Create embedding for the question
2. Search for most similar document chunks
3. Build prompt: question + relevant chunks
4. Send to LLM
5. Get grounded answer
```

### Cost considerations

- Embedding creation: one-time cost
- Embedding search: very cheap (just math)
- LLM generation: same cost as before
- **Total**: RAG adds minimal cost compared to value

---

## What makes a good RAG system?

### 1. Good chunking

- Preserve semantic coherence
- Not too small (loses context)
- Not too large (loses precision)

### 2. Good retrieval

- Accurate embeddings
- Relevant similarity threshold
- Consider metadata (date, source, author)

### 3. Good prompt engineering

- Clear instructions for using context
- Explicit citation requirements
- Handling cases with no relevant documents

### 4. Evaluation

- Are retrieved documents actually relevant?
- Does the answer use the documents correctly?
- Can users verify the sources?

---

## Common patterns you'll implement later

### Pattern 1: Semantic search

```
User searches: "data quality issues"
System finds documents about:
- "survey accuracy problems"
- "measurement error challenges"
- "validation inconsistencies"
```

### Pattern 2: Question answering
```
User asks: "What sampling method did Project X use?"
System:
1. Finds Project X documentation
2. Locates sampling methodology section
3. Generates answer with citation
```

### Pattern 3: Theme discovery
```
System:
1. Embeds all interview responses
2. Clusters similar responses
3. Uses LLM to label each cluster
4. Identifies common themes
```

---

## Limitations and considerations

### Technical limitations
- Embedding quality varies by domain/language
- Similarity search isn't perfect
- Chunking decisions affect results
- Need to manage stale documents

### Practical considerations
- Setup time for initial embedding
- Storage for vector data
- Re-embedding when documents change
- API costs for embeddings

### When RAG isn't the answer
- Very small document sets (just use long context)
- Real-time changing information (embeddings are static)
- Highly specialized domains (may need fine-tuning)
- When you need model to follow strict formats

---

## What's next

Now that you understand the theory:

**This afternoon, Session 03:**
- Implement qualitative coding workflow using embeddings
- See clustering and theme identification in practice

**This afternoon, Session 04:**
- Build an internal knowledge chatbot using RAG
- Work with real project documents
- Implement the full retrieval-augmentation-generation pipeline

Both sessions will use the concepts from today.

---

## Key takeaways

1. **Embeddings** turn text into searchable numbers that capture meaning
2. **RAG** connects LLMs to your specific data through search
3. **Use RAG** when you need LLMs to work with large, specific, or changing knowledge bases
4. **The pattern** is: retrieve relevant content, add to prompt, generate grounded answer

---

## Resources for deeper learning

- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Understanding vector embeddings: https://www.pinecone.io/learn/vector-embeddings/
- RAG explained: https://www.anthropic.com/index/contextual-retrieval
- Chunking strategies: [add link]