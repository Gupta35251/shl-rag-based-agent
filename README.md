# SHL Assessment Recommender


A conversational AI agent that helps hiring managers find the right SHL assessments through natural dialogue — built with FastAPI, FAISS, and Google Gemini.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run server
uvicorn rag_utility:app --reload
```

Open `http://127.0.0.1:8000/docs` to test.

---

## How It Works

```
User message → FAISS retrieves top-10 matching assessments → Gemini generates reply → Structured JSON response
```

1. On startup — catalog.json is embedded with `sentence-transformers` and stored in FAISS
2. On each request — all user messages combined → FAISS similarity search → top-10 retrieved
3. Retrieved items + full conversation history → sent to Gemini
4. Gemini returns structured JSON → parsed → returned as API response

---

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need a test for a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "Knowledge & Skills"}
  ],
  "end_of_conversation": false
}
```

---

## Agent Behaviors

| Behavior | When | Action |
|---|---|---|
| Clarify | Vague query | Asks one question, returns `recommendations: []` |
| Recommend | Enough context | Returns 1–10 assessments with URLs |
| Refine | User adds constraints | Updates list, does not restart |
| Compare | "Difference between X and Y?" | Answers from catalog data only |
| Refuse | Off-topic / prompt injection | Politely declines |

---

## Tech Stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Store | FAISS `IndexFlatL2` |
| LLM | Google Gemini 2.5 Flash |
| LLM Framework | LangChain |

---

## Project Structure

```
├── rag_utility.py           # FastAPI app — main entry point
├── shl_product_catalog.json # SHL catalog (377 assessments)
├── requirements.txt         # Dependencies
├── .env                     # API keys (not committed)
└── README.md
```

---

## Requirements

```
fastapi
uvicorn
faiss-cpu
sentence-transformers
langchain-google-genai
python-dotenv
numpy
```

---

