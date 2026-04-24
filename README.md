# OpenCode

Claude Code Proxy → opencode.ai. Convert Anthropic /v1/messages ↔ OpenAI chat/completions.

## Requirements

- Python 3.11+

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Edit values in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCODE_PROXY` | Proxy server | `` |
| `OPENCODE_API_KEY` | API key for OpenCode | `` |
| `OPUS_MAP_MODEL` | Model for opus route | `kimi-k2.6` |
| `SONNET_MAP_MODEL` | Model for sonnet route | `glm-5.1` |
| `HAIKU_MAP_MODEL` | Model for haiku route | `minimax-m2.5` |

## Claude Code Configuration

To use this proxy with Claude Code, add these environment variables to your shell configuration:

```bash
export ANTHROPIC_API_KEY="fake-key"
export ANTHROPIC_AUTH_TOKEN="fake"
export ANTHROPIC_BASE_URL="http://localhost:4000"
```

Or add them to your `.env` file:

```env
ANTHROPIC_API_KEY=fake-key
ANTHROPIC_AUTH_TOKEN=fake
ANTHROPIC_BASE_URL=http://localhost:4000
```

**Note:** The proxy server must be running (`python opencode.py`) before using Claude Code.

## Running

```bash
python opencode.py
```

Server will start:
- **API**: http://localhost:4000
- **Web Dashboard**: http://localhost:8082

## Web Dashboard

The web interface has 2 tabs:

### Token Stats
- Overview: Input, Output, Cache, Success, Failed, Avg Duration
- Charts: Token Distribution, Token % by Model, Requests % by Model
- Detailed table by model

### Request History
- Request logs with full info: time, model, duration, tokens, status
- Filter by time: Today, 7 Days, 30 Days, Custom
- Delete history: all or by date

### Common Features
- Dark/Light theme (Dark by default)
- Auto-refresh every 5 seconds
- Time filter shared across both tabs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Proxy Anthropic format |
| POST | `/anthropic/v1/messages` | Proxy Anthropic format |
| GET | `/health` | Health check |
| GET | `/api/stats` | Token usage stats (supports `from_date`, `to_date`) |
| GET | `/api/logs` | Terminal logs |
| GET | `/api/history` | Request history from DB (supports `from_date`, `to_date`) |
| DELETE | `/api/history` | Delete history (`before` or `all=true`) |

## Advanced Configuration

For other settings, edit `config/settings.py`:

- `PORT`: API port (default 4000)
- `WEB_PORT`: Web UI port (default 8082)
- `MODELS`: Models and endpoints list (hardcoded in code)

## Keyboard shortcuts (Terminal)

- `j`/`↓`: Scroll down log
- `k`/`↑`: Scroll up log
- `g`: Go to top
- `G`: Go to bottom
- `Ctrl+C`: Exit

## Project Structure

```
.env.example         # Template environment configuration (copy to .env)
.env                 # Environment configuration (create from .env.example)
opencode.py          # Main server
config/
  settings.py        # Configuration (includes PROXY, API_KEY, ROUTES)
static/
  index.html         # Dashboard UI
  styles.css         # Styling (dark/light theme)
  app.js             # JavaScript
logs/
  requests.db        # SQLite - request history
```
