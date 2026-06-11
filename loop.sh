#!/bin/bash
# loop.sh — self-judging improvement loop until convergence
# usage: loop.sh <site_dir> [max_rounds] [target_score]
# Alternates a thinking-mode judge and a thinking-mode fixer on index.html
# until every axis >= target (default 98) or max_rounds (default 6) is hit.
set -u
DIR=${1:?usage: loop.sh <site_dir> [max_rounds] [target]}
MAX=${2:-6}
TARGET=${3:-98}
AGENT="$(cd "$(dirname "$0")" && pwd)/agent"
cd "$DIR" || exit 1

JUDGE_PROMPT='あなたは厳格なWebサイト品質審査員。read_fileで index.html を読み、bashで裏取りしながら5軸×100点で採点して。軸: ①内容の正確さ ②デザインの美しさ ③技術品質(タグ対応・構文・セマンティクス) ④レスポンシブとアクセシビリティ ⑤パフォーマンス。採点ルール: 減点には実行可能な具体的修正指示を必ず伴うこと・推測での減点(「〜か不明」等)は禁止・実際に問題が無い軸は100点。じっくり考えてから判定。最終回答: 1行目「SCORES: ①xx ②xx ③xx ④xx ⑤xx」+減点軸ごとの修正指示。'

for ((r=1; r<=MAX; r++)); do
  echo "=== ROUND $r: judge ==="
  "$AGENT" -y -t -p "$JUDGE_PROMPT" 2>"judge_r$r.err" | tee "review_loop_r$r.txt"
  scores=$(grep -m1 -o 'SCORES:.*' "review_loop_r$r.txt" | grep -o '[0-9]\+' | head -5)
  if [ -z "$scores" ]; then echo "no SCORES line — abort"; exit 2; fi
  echo "round $r scores: $(echo $scores | tr '\n' ' ')"
  low=$(echo "$scores" | sort -n | head -1)
  if [ "$low" -ge "$TARGET" ]; then
    echo "CONVERGED at round $r (min axis $low >= $TARGET)"
    exit 0
  fi
  echo "=== ROUND $r: fix (min axis $low) ==="
  "$AGENT" -y -t -p "review_loop_r$r.txt に index.html の採点レポートがある。read_fileで読み、すべての具体的修正指示を index.html に適用して。書き直しは write_file で全体を書く(120行を超えるなら write_file+cat>>追記で分割)。『修正: なし』『問題なし』の項目はスキップ。最後に主要な修正点をgrepで1回ずつ検証して報告。" 2>"fixer_r$r.err" | tail -5
done
echo "max rounds ($MAX) reached without convergence"
exit 1
