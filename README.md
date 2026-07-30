# cagent — a Claude Code-like coding agent in ~430 lines of C

A minimal coding agent that runs entirely on a local LLM. No cloud, no API key,
no framework — just libcurl, vendored [cJSON](https://github.com/DaveGamble/cJSON),
and one C file.

Works with any OpenAI-compatible `/v1/chat/completions` server
(tested with [mlx-lm](https://github.com/ml-explore/mlx-lm) serving
Qwen3.6-35B-A3B and Nemotron-Cascade-2-30B on Apple Silicon).

## teai.io で使う（クラウド・鍵なしでも動く）

ローカルLLMを立てなくても、[teai.io](https://teai.io) のホスト済みAPIでそのまま動きます:

```bash
AGENT_BASE=teai ./agent -y -p "bashでdateを実行して結果を教えて"
```

- 鍵なし = フリー枠（Nemotron固定・10 req/分 + 50 req/日/IP）。上限に達すると
  cagent がキーの取り方を案内する
- `AGENT_KEY=te_...`（teai.ioのAPIキー）で100+モデルのフルカタログ。
  利用分はアカウントのクレジットから引かれる（Free 100 / Starter 月25,000 / Pro 30,000）
- モデル未指定ならサーバ側オートルーター `teai/auto` に任せる（`-m` で明示指定も可）

Nemotron系の推論モデルが答えを `<think>` 内に書いて黙るケースは、think内テキストを
表示にフォールバックすることで対処済み。

## 声で使う（KOE — koe.live）

`-k` をつけると、エージェントの回答を [KOE](https://koe.live) の実声TTSが読み上げ、
REPLで `v` + Enter と打つと**マイクで話した内容がそのままプロンプトになる**（STTもKOE）:

```bash
AGENT_BASE=teai ./agent -k          # 回答を声で聞く / 'v' で声で指示
```

| env | 意味 |
|---|---|
| `KOE_VOICE` | 読み上げの声ID（既定 `kentaro`＝本人同意済みの公開サンプル） |
| `KOE_KEY` | koe.liveのトークン（`X-Koe-Admin`）。**自分の声のその場登録**に必要 |
| `KOE_BASE` | 既定 `https://koe.live` |

自分の声で読み上げたいときは、その場で登録できる（お題を1文読むだけ・要 `KOE_KEY`）:

```bash
KOE_KEY=... ./agent --koe-enroll        # 録音→本人ライブネス検証→声ID発行
export KOE_VOICE=<発行されたID>
```

鍵なしでも読み上げ（公開ボイス）と音声操作は動く。声の登録だけはKOEの背骨
（本人同意ゲート＝お題をその場で読めた人しか登録できない）に従いログイン/キー必須。
録音は ffmpeg / sox / arecord のどれかを使う。再生は afplay / mpg123 / ffplay。

## What it does

The whole trick of a coding agent is one loop:

```
call LLM → got tool_calls? → run them → send results back → repeat
                └─ no → print final answer
```

cagent implements that loop with 4 tools — `bash`, `read_file`, `write_file`,
`edit_file` — and asks permission (`[y/N]`) before anything destructive.
That's enough to write files, compile programs, fix its own JSON mistakes,
and even design a building (we made it model the Barcelona Pavilion on
[bim.house](https://bim.house) via MCP-over-curl, self-judged in a loop).

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/yukihamada/cagent/master/install.sh | bash
cagent -b teai -y -p "hello"       # 鍵なしでそのまま試せる
```

ソースを取ってきて `make` するだけのインストーラ（配布バイナリなし・依存は libcurl のみ）。

## Build & run

```bash
make                       # cc -O2 -Wall -Wextra -o agent agent.c cJSON.c -lcurl

./agent                    # interactive REPL
./agent -y -p "task..."    # one-shot, auto-approved (for scripts/evals)
./eval.sh                  # 7-task eval harness (PASS/FAIL + timing)
```

| flag / env | default | meaning |
|---|---|---|
| `-b` / `AGENT_BASE` | `http://127.0.0.1:8780` | OpenAI-compatible server (`teai` = https://api.teai.io) |
| `-m` / `AGENT_MODEL` | `mlx-community/Qwen3.6-35B-A3B-4bit` | model id (teai.io なら既定 `teai/auto`) |
| `AGENT_KEY` | — | Bearer トークン (teai.io の `te_...` APIキー等) |
| `-p` | — | one-shot prompt (exit after) |
| `-y` | off | auto-approve tool calls |
| `-t` / `AGENT_THINK` | off | thinking mode — the model reasons before acting (~5-10x tokens, much better judgment) |

For a remote LLM host: `ssh -L 8780:127.0.0.1:8780 user@llm-host`, then run as usual.

## Hard-won notes (read if you build your own)

- **mlx-lm ≥ 0.31 parses `<tool_call>` server-side.** A prompt-only text
  protocol gets swallowed — the response comes back with an empty `message`.
  You MUST pass `tools` in the request and read `message.tool_calls`.
- **Never truncate UTF-8 mid-sequence.** Tool output is capped (64KB); cutting
  a multibyte char in half produces invalid JSON that crashes
  `mlx_lm.server`'s request handler. `utf8_trim()` backs off to a boundary.
- **`finish_reason: "length"` needs recovery.** A model that hits max_tokens
  mid-tool-call has done nothing; tell it explicitly to continue in smaller
  chunks or it will happily claim success.
- **Qwen3.6 on mlx needs `chat_template_kwargs: {enable_thinking: false}`.**
- Trim old tool results from history (keep the recent tail verbatim) or long
  sessions eat the context window.
- Local-LLM agents fail differently than frontier ones: they loop on the same
  verification command, invent API schemas, and mangle JSON-RPC plumbing.
  Give them pre-verified schemas and split big writes into ≤100-line chunks.
- **Thinking mode is worth the tokens for judging.** The same file scored
  100/100/100/100/100 by a non-thinking judge scored 90/75/70/65/95 with
  thinking on — and the thinking judge found a *real functional bug*
  (a `<script>` running before its target element existed) that two
  non-thinking judges had missed.

## Files

| file | what |
|---|---|
| `agent.c` | the agent (~430 lines) |
| `cJSON.c/.h` | vendored JSON library |
| `eval.sh` | 7-task eval harness in a sandbox |
| `site/` | self-introduction sites the agent built about itself, self-judged to (near) 100/100 |

## License

MIT
