# Directory Agent - AI Agents Assessment

This is a small Python agent that answers questions about the supplied fictional company directory. It uses a local Qwen model through Ollama to decide which tools to call. The final response is rendered from the returned records so the agent does not turn a plausible model sentence into an unsupported fact.

## How to run it

The project uses Python's standard library only. It was tested with Ollama and `qwen2.5:3b`.

```bash
# Install and start Ollama if it is not already running.
ollama pull qwen2.5:3b
ollama serve

# In another terminal, from this repository:
python agent.py "Who is the CTO of Nordwind Analytics?"

# Or run the five assessment questions:
python agent.py
```

If a different local model is installed, set it before running:

```bash
export OLLAMA_MODEL="your-model-name"
python agent.py
```

Run the small test suite with:

```bash
python -m unittest discover -s tests -v
```

## Agent output

These are actual runs with `qwen2.5:3b`. The trace is printed before each answer.

### Q1. Who is the CTO of Nordwind Analytics?

```text
TOOL search_directory({"query": "CTO Nordwind Analytics"})
Companies: Nordwind Analytics (Data Analytics, Berlin, domain: nordwind-analytics.io)
People: Marta Ibanez — CTO at Nordwind Analytics
```

### Q2. Which people work at companies in the Healthcare industry?

```text
TOOL search_directory({"query": "industry:Healthcare"})
People: Grace Okafor — Chief Medical Officer at Verdant Health;
Lena Kovac — Board Member at Verdant Health;
Lena Kovac — CFO at Aurora BioLabs;
Daniel Osei — Research Director at Aurora BioLabs
```

### Q3. What is the website domain of Cobalt Marine?

```text
TOOL search_directory({"query": "Cobalt Marine"})
TOOL enrich_company({"company_name": "Cobalt Marine"})
Enrichment candidates: cobaltmarine.nl (confidence 0.97, business_registry)
```

### Q4. What is the website domain of Helios Data?

```text
TOOL search_directory({"query": "Helios Data"})
TOOL enrich_company({"company_name": "Helios Data"})
Enrichment candidates: heliosdata.com (confidence 0.72, web_search);
helios-data.ai (confidence 0.68, web_search)

The data is ambiguous, so the agent returns both candidates instead of choosing one.
```

### Q5. What was Sable Security's revenue in 2025?

```text
TOOL search_directory({"query": "Sable Security revenue 2025"})
Companies: Sable Security (Cybersecurity, Tel Aviv, domain: sablesec.com)
People: Avi Shulman — CISO at Sable Security
The retrieved data does not contain a revenue field.
```

## Technical decisions

### 1. Why a custom loop instead of a framework?

I used a small custom loop on top of Ollama's chat API. The assessment is about understanding tool calls and stopping conditions, and this keeps those pieces in one file: `call_ollama()` sends the tool schema, `answer()` handles tool calls, and `MAX_ITERATIONS` stops the loop.

LangChain or LangGraph would reduce some plumbing and offer more built-in patterns for larger workflows. The trade-off is more dependencies and more framework behavior to explain. For two tools and a single turn, I preferred the smaller surface area.

### 2. How does the agent decide which tool to call?

The model receives both tool definitions in `TOOLS`. It responds with `tool_calls`, and `answer()` dispatches the requested function by name. The system prompt asks it to inspect the directory first and decide whether enrichment is needed after seeing the directory result. For the domain questions, it chose `search_directory` and then `enrich_company` because the CSV had no domain.

The code does not route individual questions with an `if/else`. The only guard is that if the model tries to answer without inspecting the directory, it gets one reminder to choose a tool. This is a general grounding rule, not a question-specific route.

### 3. How does it stop looping or burning tokens?

`MAX_ITERATIONS = 4`. Each iteration either receives tool calls and feeds results back to the model, or returns the answer. If it reaches the limit, it stops with an explicit message. Tool failures also raise a clear error when Ollama cannot be reached.

### 4. Why keyword/structured retrieval instead of embeddings?

There are only 31 directory rows and the fields are structured: company name, industry, title, and city. Exact company-name matching plus keyword search is deterministic, fast, and easy to inspect. Embeddings would add a model, index, and similarity threshold without improving these five queries.

At 500,000 rows, I would move the data to a database with normalized company/person tables and indexes for company name, industry, and title. I would add hybrid retrieval: structured filters for known fields and embeddings/full-text search for fuzzy natural-language requests.

### 5. What would change for AWS Bedrock?

The retrieval tools would stay the same. I would replace `call_ollama()` with a Bedrock Runtime client call, use the selected model's tool-use message format, and load AWS credentials through the normal SDK credential chain. The loop, tool dispatch, trace, maximum iteration rule, and grounding renderer would not need to change.

### 6. When would a graph database fit Q2 better?

Q2 joins people to companies through the company relationship and filters companies by industry. A graph database becomes useful when those relationships become multi-hop or exploratory: for example, finding people connected to suppliers of healthcare companies, or following shared board memberships across organizations. For this small direct join, CSV/relational data is simpler and clearer than either a graph or vector search.

### Grounding choice

Qwen 3B can reliably select tools for this task but can occasionally paraphrase a table incorrectly, especially when a person appears at two companies. For that reason, the agent prints the tool trace and renders the final result from the exact records returned by the tools. The LLM still decides the tool sequence; the renderer prevents unsupported facts from being introduced after retrieval.

## What I would do differently with more time

1. Add a small evaluation script with expected answers and tool-call expectations for the five questions.
2. Support follow-up questions by preserving the conversation and the already-retrieved evidence.
3. Add a relational database backend and hybrid retrieval for a much larger directory.
4. Add structured activity logs with timing and model/tool errors rather than printing traces to stdout.

## Difficulties I encountered

The hardest part was grounding the final answer, not reading CSV files. A small local model could call the correct retrieval tool but sometimes combine two matching-looking rows in prose. I kept the LLM responsible for tool selection, then made the final output evidence-based. That made the ambiguous Helios result and the missing Sable revenue answer explicit instead of guessed.
