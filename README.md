# Kenyan Constitutional Triage

A local lab for answering a described situation with the Constitution of Kenya (2010). Everything runs on this machine through [Ollama](https://ollama.com). There is no paid classifier and no PyTorch install.

The system does three things, in order:

1. **Triage.** `gemma3:1b` labels the situation: how urgent it is, which part of the Constitution it touches, and how likely it is that the police are needed.
2. **Retrieval.** The question is embedded with `nomic-embed-text` and matched against chunks of the Constitution stored in Chroma.
3. **Guidance.** `gemma3:1b` writes a short directive from those labels and the retrieved passages.

This is a study aid for finding relevant constitutional text. It is not a lawyer, and it does not replace emergency services.

## What is in this folder

| File | Role |
| --- | --- |
| `The_Constitution_of_Kenya_2010.pdf` | Source document |
| `ingest.py` | Reads the PDF, splits it, embeds the chunks, and saves them |
| `app.py` | Asks for a situation, triages it, retrieves passages, and writes the directive |
| `requirements.txt` | Python packages: Chroma, pypdf, the Ollama client, and Rich |

`ingest.py` stores the index in `chroma_kenyan_law_db/`. That folder is generated on each machine and is listed in `.gitignore`.

## Models

Both models are open weights served by Ollama.

| Step | Model | Why this one |
| --- | --- | --- |
| Embed the Constitution and the question | `nomic-embed-text:latest` | Built to produce vectors. A chat model cannot do this job. |
| Triage and the written directive | `gemma3:1b` | Small enough to run on a laptop with about 8 GB of RAM. |

The same embedding model has to be used when the index is built and when a question is searched. `app.py` and `ingest.py` both use `nomic-embed-text:latest`.

`gemma3:1b` is used for both chat steps so only one chat model sits in memory. Larger local models such as `llama3.2` or `gemma4:e2b` are optional later, if the machine has the free RAM.

## Setup

Install [Ollama](https://ollama.com), then pull the two models:

```bash
ollama pull nomic-embed-text
ollama pull gemma3:1b
```

From this folder, create a virtual environment and install the Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Leave the Constitution PDF next to `ingest.py`, under the name `The_Constitution_of_Kenya_2010.pdf`.

## Build the index

Ollama must be running.

```bash
python ingest.py
```

That script extracts the PDF text, cuts it into windows of 1,000 characters with a 200-character overlap, embeds each window, and stores  the vectors in a Chroma collection named `kenyan_constitution`. A full run on this document produces about 413 chunks.

Run `ingest.py` again any time the PDF or the chunking changes. It replaces the old collection.

## Ask a question

```bash
python app.py
```

Type a situation in plain language. The program then prints:

- **Urgency:** `immediate_emergency`, `urgent_intervention`, or `legal_guidance`, plus a confidence from 0 to 1
- **Domain:** `bill_of_rights`, `children_rights`, or `general_state_obligations`
- **Police:** a score from 0 to 1
- A short directive that includes safety steps, constitutional passages taken from the retrieved chunks, and the contacts 1195 (GBV), 116 (Childline Kenya), and 999 / 112 (police)

The triage reply is forced into those labels with an Ollama JSON schema. The confidence and police figures are the chat model's own estimates, not a measured probability.

## Lab notes

- Chunks are fixed lengths of characters. A cut can fall in the middle of an article, so a retrieved passage may not contain the article number that belongs to it.
- `gemma3:1b` can pick a softer urgency label than the facts deserve. Read the retrieved constitutional text, not only the label.
- Do not install PyTorch or a Laya checkpoint on a machine that is short of disk or RAM. This lab stays on the Ollama models above.
- Keep the virtual environment and `chroma_kenyan_law_db/` out of git. The `.gitignore` in this folder already does that.
