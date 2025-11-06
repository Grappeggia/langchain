# LangChain FastAPI with Vercel Zero Config

A basic LangChain application built with FastAPI and optimized for Vercel's zero config deployment.

## Features

- ✨ FastAPI with async support
- 🔗 LangChain integration with OpenAI models
- 🚀 Vercel zero config FastAPI support
- 📝 Full API documentation with Swagger UI
- 🔄 CORS enabled for frontend integration
- 📦 Environment variable management with `.env`

## Project Structure

```
.
├── api/
│   └── index.py              # Main FastAPI application
├── requirements.txt          # Python dependencies
├── vercel.json              # Vercel configuration
├── README.md                # This file
└── .env                     # Environment variables (create locally)
```

## Setup

### 1. Prerequisites

- Python 3.11+
- Node.js (for Vercel CLI, optional)
- OpenAI API key

### 2. Local Development

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Run the development server:

```bash
cd api
python index.py
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs`

### 3. Vercel Deployment

Install the Vercel CLI:

```bash
npm install -g vercel
```

Deploy to Vercel:

```bash
vercel
```

Set environment variables in Vercel dashboard or via CLI:

```bash
vercel env add OPENAI_API_KEY
```

## API Endpoints

### Health Check

```bash
GET /
GET /health
```

### Query Endpoint

```bash
POST /api/query

Request:
{
  "query": "What is machine learning?",
  "system_prompt": "You are a helpful assistant.",
  "temperature": 0.7
}

Response:
{
  "query": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence...",
  "model": "gpt-3.5-turbo"
}
```

### Chat Endpoint

```bash
POST /api/chat

Request:
{
  "query": "Hello, how are you?",
  "system_prompt": "You are a friendly chatbot.",
  "temperature": 0.8
}

Response:
{
  "message": "I'm doing well, thank you for asking!",
  "model": "gpt-3.5-turbo"
}
```

### List Available Models

```bash
GET /api/models

Response:
{
  "models": [
    {
      "name": "gpt-3.5-turbo",
      "description": "Fast and efficient model",
      "context_window": 4096
    },
    {
      "name": "gpt-4",
      "description": "More capable model (requires access)",
      "context_window": 8192
    }
  ]
}
```

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Query the model
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LangChain?",
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7
  }'

# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me a joke",
    "system_prompt": "You are a comedian.",
    "temperature": 0.9
  }'

# List models
curl http://localhost:8000/api/models
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Vercel Configuration

The `vercel.json` file configures:

- **Runtime**: Python 3.11
- **Memory**: 1024 MB per function
- **Max Duration**: 30 seconds per request

Adjust these values as needed for your use case.

## Next Steps

### Extend the Application

1. **Add more LangChain chains**: Implement question-answering, summarization, or other chains
2. **Add memory**: Implement conversation history with LangChain memory
3. **Add tools**: Integrate external APIs and tools
4. **Error handling**: Enhance error messages and logging
5. **Authentication**: Add API key authentication if needed

### Example: Adding a Summarization Endpoint

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

@app.post("/api/summarize")
async def summarize(request: QueryRequest):
    llm = get_llm()
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.split_documents([request.query])
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return {"summary": summary}
```

## Troubleshooting

### "OPENAI_API_KEY not set" error

Ensure your `.env` file is in the project root and contains a valid OpenAI API key.

### 502 Bad Gateway on Vercel

This typically means:
1. The function is exceeding the 30-second timeout
2. There's an environment variable missing
3. The OpenAI API is unreachable

Check the Vercel function logs for more details.

### Local development issues

Make sure you're running the script from the correct directory:

```bash
cd api
python index.py
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Vercel Python Support](https://vercel.com/docs/concepts/functions/serverless-functions/python)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## License

MIT

