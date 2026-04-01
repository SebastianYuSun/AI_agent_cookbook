# AI Agent Cookbook

A personal lab for experimenting with LLM APIs and agent architectures across
AWS Bedrock, Google Cloud, and OpenAI. Each experiment lives in its own
directory; shared infrastructure lives in `utils/`.

---

## Project Structure

```
AI_agent_cookbook/
├── utils/
│   ├── base.py           # Abstract adapter interface
│   ├── config.py         # .env loader + helpers
│   ├── openai_client.py  # OpenAI adapter
│   ├── aws_claude.py     # AWS Bedrock (Claude) adapter
│   └── google_cloud.py   # Google Gemini adapter
├── tests/
│   ├── test_openai_client.py
│   ├── test_aws_claude.py
│   └── test_google_cloud.py
├── config/               # Local secrets (gitignored)
├── .env.example          # Template — copy to .env
├── .github/workflows/
│   └── ci.yml            # GitHub Actions: run tests on every PR
├── requirements.txt
├── requirements-dev.txt
└── Makefile
```

---

## Quick Start

### 1. Set up the environment

```bash
make setup
source .venv/bin/activate
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your real credentials
```

### 3. Use an adapter

```python
from utils.openai_client import OpenAIAdapter
from utils.aws_claude import AWSClaudeAdapter
from utils.google_cloud import GoogleCloudAdapter

# All adapters share the same interface
openai   = OpenAIAdapter()
bedrock  = AWSClaudeAdapter()
google   = GoogleCloudAdapter()

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]

for adapter in [openai, bedrock, google]:
    print(type(adapter).__name__, "→", adapter.chat(messages))

# Or use the one-liner shortcut:
print(openai.complete("What is 2 + 2?"))
```

---

## Running Tests

```bash
make test          # run tests + coverage report
make lint          # ruff linter
make fmt           # black formatter
```

Tests use mocks — **no real API keys are needed** to run them.

---

## Enforcing Tests Before Merge (Branch Protection)

The CI workflow (`.github/workflows/ci.yml`) runs automatically on every pull
request. To block merges when tests fail:

1. Go to your repo on GitHub → **Settings → Branches**
2. Click **Add rule** for the `main` branch
3. Enable **Require status checks to pass before merging**
4. Search for and add the `test` check
5. Optionally enable **Require branches to be up to date before merging**

After this, GitHub will block any PR where CI fails.

---

## Provider Configuration Reference

| Variable | Provider | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | yes |
| `OPENAI_MODEL` | OpenAI | no (default: `gpt-4o`) |
| `AWS_ACCESS_KEY_ID` | AWS Bedrock | yes* |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock | yes* |
| `AWS_DEFAULT_REGION` | AWS Bedrock | no (default: `us-east-1`) |
| `AWS_BEDROCK_MODEL_ID` | AWS Bedrock | no (default: Claude 3.5 Sonnet) |
| `GOOGLE_API_KEY` | Google Gemini | yes |
| `GOOGLE_MODEL` | Google Gemini | no (default: `gemini-1.5-pro`) |

*Not required when running on EC2/ECS with an attached IAM role.
