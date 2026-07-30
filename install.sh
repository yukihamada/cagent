#!/usr/bin/env bash
# cagent installer — builds from source (no binaries to trust, ~1s compile).
#   curl -fsSL https://raw.githubusercontent.com/yukihamada/cagent/master/install.sh | bash
#
# Env:
#   PREFIX  install dir (default: /usr/local/bin, or ~/.local/bin if not writable)
set -euo pipefail

REPO="https://github.com/yukihamada/cagent"
PREFIX="${PREFIX:-}"

say() { printf '\033[36m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || die "no C compiler. macOS: xcode-select --install / Debian: apt install build-essential"
command -v git >/dev/null 2>&1 || die "git not found"

# libcurl headers are the only dependency
if ! printf '#include <curl/curl.h>\nint main(void){return 0;}\n' | cc -x c - -lcurl -o /dev/null 2>/dev/null; then
  die "libcurl dev headers not found. macOS: brew install curl / Debian: apt install libcurl4-openssl-dev"
fi

if [ -z "$PREFIX" ]; then
  if [ -w /usr/local/bin ] 2>/dev/null; then PREFIX=/usr/local/bin; else PREFIX="$HOME/.local/bin"; fi
fi
mkdir -p "$PREFIX"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
say "→ fetching cagent…"
git clone --depth 1 -q "$REPO" "$TMP/cagent"
say "→ building…"
make -C "$TMP/cagent" >/dev/null
install -m 0755 "$TMP/cagent/agent" "$PREFIX/cagent"

say "✅ installed: $PREFIX/cagent"
case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *) printf '\033[33mnote:\033[0m %s is not in PATH — add:  export PATH="%s:$PATH"\n' "$PREFIX" "$PREFIX" ;;
esac

cat <<'EOF'

使い方:
  cagent -b teai -y -p "タスクを書く"     # クラウド(teai.io)・鍵なしで試せる
  cagent -b teai -k                        # 声で聞く / REPLで v と打てば声で指示
  cagent                                   # ローカルLLM (http://127.0.0.1:8780)

無料枠の先へ（100+モデル・クレジット課金）:
  1. https://teai.io で登録
  2. APIキー(te_...)を発行
  3. export AGENT_KEY=te_...

自分の声で読み上げ（koe.live）:
  KOE_KEY=... cagent --koe-enroll     # お題を1文読むだけで声を登録
  export KOE_VOICE=<発行されたID>
EOF
