#!/bin/bash
# cagent eval harness — runs tasks in a sandbox and scores PASS/FAIL
set -u
AGENT=${AGENT:-$HOME/workspace/cagent/agent}
export AGENT_BASE=${AGENT_BASE:-http://127.0.0.1:18780}
export AGENT_MODEL=${AGENT_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}
SB=$(mktemp -d /tmp/cagent_eval.XXXXXX)
cd "$SB" || exit 1
echo "sandbox: $SB"
echo "model:   $AGENT_MODEL"
pass=0; fail=0; results=""
out=""

run_task() { # name prompt check_expr
  local name="$1" prompt="$2" check="$3"
  local t0 t1 rc
  t0=$(date +%s)
  out=$(timeout 240 "$AGENT" -y -p "$prompt" 2>llm.log)
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ] && eval "$check"; then
    pass=$((pass+1)); results+="PASS $((t1-t0))s  $name\n"
  else
    fail=$((fail+1))
    results+="FAIL $((t1-t0))s  $name (rc=$rc)\n   out: $(echo "$out" | tail -1)\n"
  fi
}

run_task "T1 write file" \
  "hello.txt というファイルを作って、中身を konnichiwa という1行だけにして" \
  '[ "$(cat hello.txt 2>/dev/null)" = "konnichiwa" ]'

printf 'name = OLD_NAME\nversion = 1\n' > config.ini
run_task "T2 edit file" \
  "config.ini の OLD_NAME を NEW_NAME に書き換えて" \
  'grep -q "name = NEW_NAME" config.ini && grep -q "version = 1" config.ini'

printf 'apple\nbanana\ncherry\n' > fruits.txt
run_task "T3 read & answer" \
  "fruits.txt を読んで、2行目の単語だけを最終回答にして" \
  'echo "$out" | grep -qi banana'

run_task "T4 write+compile+run C" \
  "HELLO_C とだけ出力する hello.c を書いて、cc でコンパイルして実行し、出力を確認して" \
  'cc hello.c -o /tmp/cagent_hx 2>/dev/null && [ "$(/tmp/cagent_hx)" = "HELLO_C" ]'

mkdir -p sub; touch sub/a.log sub/b.log sub/c.txt
run_task "T5 count via bash" \
  "sub ディレクトリにある .log ファイルの個数を数えて、最終回答は数字だけにして" \
  'echo "$out" | grep -qE "(^|[^0-9])2([^0-9]|$)"'

run_task "T6 plain answer (no tool)" \
  "ツールを使わずに 7*6 の答えを数字だけで答えて" \
  'echo "$out" | grep -q 42'

run_task "T7 multi-step fix" \
  "broken.py を作らず、まず sub/c.txt に PING と書き、次にそれを読んで内容が PING なら最終回答を PONG にして" \
  'echo "$out" | grep -q PONG && [ "$(cat sub/c.txt)" = "PING" ]'

echo ""
echo "=== RESULTS ($AGENT_MODEL) ==="
printf "%b" "$results"
echo "pass=$pass fail=$fail"
