import json
import os

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama
from rich.console import Console
from rich.panel import Panel

console = Console()
DB_DIR = "./chroma_kenyan_law_db"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "gemma3:1b"

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "urgency": {
            "type": "string",
            "enum": [
                "immediate_emergency",
                "urgent_intervention",
                "legal_guidance",
            ],
        },
        "urgency_confidence": {"type": "number"},
        "domain": {
            "type": "string",
            "enum": [
                "bill_of_rights",
                "children_rights",
                "general_state_obligations",
            ],
        },
        "requires_police": {"type": "number"},
    },
    "required": [
        "urgency",
        "urgency_confidence",
        "domain",
        "requires_police",
    ],
}


def _clamp_unit(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def get_vector_store():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(
        name="kenyan_constitution",
        embedding_function=OllamaEmbeddingFunction(
            url=OLLAMA_URL,
            model_name=EMBEDDING_MODEL,
        ),
    )


def evaluate_scenario(scenario: str):
    """Label urgency, legal domain, and police need with the local chat model."""
    prompt = f"""Classify this situation for Kenyan legal-aid routing.

urgency:
- immediate_emergency: life-threatening danger needing police or medical help now
- urgent_intervention: active abuse or a severe violation needing protection within 24 hours
- legal_guidance: a rights issue or past grievance needing legal aid or counseling

domain:
- bill_of_rights: dignity, freedom and security of the person, or equality
- children_rights: a child's welfare, abuse, or neglect
- general_state_obligations: public administration or governance duties

requires_police: probability from 0 to 1 that an active criminal offense needs immediate police.
urgency_confidence: probability from 0 to 1 that the urgency label is right.

Situation:
{scenario}
"""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=TRIAGE_SCHEMA,
        options={"temperature": 0},
    )
    data = json.loads(response["message"]["content"])
    return {
        "urgency": data["urgency"],
        "urgency_confidence": _clamp_unit(data["urgency_confidence"]),
        "domain": data["domain"],
        "requires_police": _clamp_unit(data["requires_police"]),
    }


def query_constitution_rag(scenario: str):
    """Retrieve the closest Constitution chunks."""
    if not os.path.exists(DB_DIR):
        return "Local vector database not found. Please run ingest.py first."
    collection = get_vector_store()
    results = collection.query(query_texts=[scenario], n_results=3)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)


def generate_directive(scenario: str, triage: dict, legal_context: str):
    """Write the guidance from the labels and retrieved Constitution text."""
    prompt = f"""
    You are an expert Kenyan legal aid assistant and human rights advisor.

    System Triage Metrics:
    - Urgency Level: {triage["urgency"]}
    - Legal Domain: {triage["domain"]}
    - Police Intervention Required: {triage["requires_police"]:.2f}

    Retrieved Constitutional Context:
    {legal_context}

    User Scenario: "{scenario}"

    Provide a structured, calm, and practical terminal directive containing:
    1. Immediate Safety & Action Steps.
    2. Applicable Constitutional Articles or Legal Sections based on the retrieved context.
    3. Critical Kenyan Emergency Contacts (National GBV Hotline: 1195, Childline Kenya: 116, Police Emergency: 999 / 112).
    """
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You provide precise, calm, and accurate legal navigation steps strictly grounded in Kenyan law.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Kenyan Constitutional Triage & RAG System[/bold cyan]\n"
            "[dim]Ollama gemma3:1b triage + ChromaDB (nomic-embed-text) + gemma3:1b[/dim]"
        )
    )

    scenario = console.input("\n[bold yellow]Describe the situation or scenario:[/bold yellow]\n> ")
    if not scenario.strip():
        console.print("[red]Scenario cannot be empty.[/red]")
        return

    with console.status("[bold green]Classifying urgency and domain locally...[/bold green]"):
        triage = evaluate_scenario(scenario)

    console.print("\n[bold green]✓ Local triage:[/bold green]")
    console.print(
        f" • Urgency: [cyan]{triage['urgency']}[/cyan] "
        f"(Confidence: {triage['urgency_confidence']:.2f})"
    )
    console.print(f" • Domain: [cyan]{triage['domain']}[/cyan]")
    console.print(
        f" • Police Intervention Needed: [cyan]{triage['requires_police']:.2f}[/cyan]"
    )

    with console.status("[bold green]Querying Local Constitutional Vector DB (RAG)...[/bold green]"):
        legal_context = query_constitution_rag(scenario)

    with console.status("[bold green]Consulting local Ollama model for legal synthesis...[/bold green]"):
        directive = generate_directive(scenario, triage, legal_context)

    console.print(
        Panel(
            directive,
            title="[bold red]Actionable Legal Directive & Guidance[/bold red]",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
