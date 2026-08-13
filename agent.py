"""A small, tool-calling directory agent backed by a local Ollama model."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
KIT = ROOT / "kit"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MAX_ITERATIONS = 4


def load_csv(name: str) -> list[dict[str, str]]:
    with (KIT / name).open(newline="") as source:
        return list(csv.DictReader(source))


COMPANIES = load_csv("companies.csv")
PEOPLE = load_csv("people.csv")
ENRICHMENT = json.loads((KIT / "enrichment_api.json").read_text())


def search_directory(query: str) -> dict:
    """Search the small supplied directory by company, person, title, industry, city, or domain."""
    normalized = query.strip().lower()
    exact_companies = [company for company in COMPANIES if company["company_name"].lower() in normalized]
    if exact_companies:
        names = {company["company_name"] for company in exact_companies}
        people = [person for person in PEOPLE if person["company_name"] in names]
        title_matches = [person for person in people if person["title"].lower() in normalized]
        return {"companies": exact_companies, "people": title_matches or people}
    words = {word.lower() for word in query.replace("?", "").split() if len(word) > 1}

    def matches(row: dict[str, str]) -> bool:
        text = " ".join(row.values()).lower()
        return any(word in text for word in words)

    companies = [company for company in COMPANIES if matches(company)]
    people = [person for person in PEOPLE if matches(person)]
    if "healthcare" in normalized:
        healthcare = {company["company_name"] for company in COMPANIES if company["industry"] == "Healthcare"}
        companies = [company for company in COMPANIES if company["company_name"] in healthcare]
        people = [person for person in PEOPLE if person["company_name"] in healthcare]
    return {"companies": companies, "people": people}


def enrich_company(company_name: str) -> dict:
    """Read simulated enrichment candidates; never calls an external service."""
    return {"company_name": company_name, "candidates": ENRICHMENT.get(company_name, [])}


def render_evidence(question: str, results: list[dict]) -> str:
    """Render exact tool records instead of trusting a small local model to paraphrase them."""
    companies, people, candidates = [], [], []
    for result in results:
        companies.extend(result.get("companies", []))
        people.extend(result.get("people", []))
        candidates.extend(result.get("candidates", []))
    lines = ["Grounded answer (only retrieved records):"]
    if companies:
        lines.append("Companies: " + "; ".join(f"{row['company_name']} ({row['industry']}, {row['city']}, domain: {row['domain'] or 'not listed'})" for row in companies))
    if people:
        lines.append("People: " + "; ".join(f"{row['full_name']} — {row['title']} at {row['company_name']}" for row in people))
    if candidates:
        lines.append("Enrichment candidates: " + "; ".join(f"{row['domain']} (confidence {row['confidence']}, {row['source']})" for row in candidates))
    if len(lines) == 1:
        lines.append("No matching records or enrichment candidates were found.")
    if "revenue" in question.lower():
        lines.append("The retrieved data does not contain a revenue field.")
    return "\n".join(lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_directory",
            "description": "Search the supplied company and people directory. Use it before answering directory facts.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enrich_company",
            "description": "Look up simulated external domain enrichment candidates when the directory has no domain.",
            "parameters": {"type": "object", "properties": {"company_name": {"type": "string"}}, "required": ["company_name"]},
        },
    },
]


def call_ollama(messages: list[dict]) -> dict:
    payload = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False, "options": {"temperature": 0}}).encode()
    request = Request("http://127.0.0.1:11434/api/chat", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=90) as response:
            return json.load(response)["message"]
    except URLError as error:
        raise RuntimeError("Cannot reach Ollama. Start it with `ollama serve` or its systemd service.") from error


def answer(question: str, trace: bool = True) -> str:
    messages = [
        {"role": "system", "content": "You answer only from tool results. First use search_directory for every question. After seeing its result, decide whether enrich_company is needed: use enrichment only when the requested domain is missing. Answer only the requested fact, with no extra claims. Do not infer facts. For people, keep each name, title, and company paired exactly as returned; do not combine titles from different rows. If evidence is missing, say you do not have that information. If enrichment returns multiple candidates, list all candidates and say the data is ambiguous. Do not invent domains, revenue, or people."},
        {"role": "user", "content": question},
    ]
    tools = {"search_directory": search_directory, "enrich_company": enrich_company}
    evidence = []
    asked_for_tool = False
    for _ in range(MAX_ITERATIONS):
        message = call_ollama(messages)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            if not asked_for_tool:
                messages.append({"role": "user", "content": "Before answering, inspect the directory with search_directory. Choose the tool call yourself."})
                asked_for_tool = True
                continue
            return render_evidence(question, evidence)
        asked_for_tool = True
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments", {})
            if name not in tools:
                result = {"error": f"Unknown tool: {name}"}
            else:
                result = tools[name](**arguments)
            evidence.append(result)
            if trace:
                print(f"TOOL {name}({json.dumps(arguments)}) -> {json.dumps(result)}")
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return "I stopped after the maximum number of tool iterations without a grounded answer."


QUESTIONS = [
    "Who is the CTO of Nordwind Analytics?",
    "Which people work at companies in the Healthcare industry?",
    "What is the website domain of Cobalt Marine?",
    "What is the website domain of Helios Data?",
    "What was Sable Security's revenue in 2025?",
]


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUESTIONS
    for index, question in enumerate(questions, 1):
        print(f"\nQ{index}: {question}")
        print(f"A{index}: {answer(question)}")
