#!/usr/bin/env bash
# Pallama soak harness: sustained mixed load with health snapshots.
# Usage: ./scripts/soak.sh [MINUTES] [BASE_URL] [MODEL]
# Output: CSV at /tmp/pallama-soak.csv + summary. Abort if any request
# fails or any snapshot shows an unexpected state regression.

set -u
MINUTES="${1:-2}"
BASE="${2:-http://127.0.0.1:11434}"
MODEL="${3:-qwen3.5-9b:q4_k_m}"
CSV="/tmp/pallama-soak.csv"
END=$(( $(date +%s) + MINUTES * 60 ))

echo "soak: ${MINUTES}m against $BASE model=$MODEL"
echo "ts,iter,kind,status,ms,ok" > "$CSV"

snap() {
    local ts=$1
    local ps vr rss
    ps=$(curl -s -m 5 "$BASE/api/ps" 2>/dev/null || echo '{}')
    # child RSS+VRAM via pgrep/nvidia-smi (optional tools, best-effort)
    rss=$(pgrep -f "llama-server.*--alias" | head -1 | xargs -r ps -o rss= -p 2>/dev/null | awk '{printf "%.0f", $1/1024}')
    vr=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "snap ts=$ts loaded=$(echo "$ps" | jq -r '.models | length') inflight=$(echo "$ps" | jq -r '[.models[].pallama_in_flight] | add // 0') child_rss_mb=${rss:-na} gpu_used_mb=${vr:-na} evictions=$(curl -s -m 5 "$BASE/metrics" | awk '/^pallama_evictions_total/ {print $2}')"
}

i=0; fail=0
while [ "$(date +%s)" -lt "$END" ]; do
    i=$((i+1))
    for kind in chat_stream chat_batch openai_models; do
        ts=$(date +%s)
        case $kind in
            chat_stream)
                ms=$(curl -s -N -m 120 -o /dev/null -w '%{time_total}|%{http_code}' "$BASE/api/chat" \
                    -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"say hi iter $i\"}],\"options\":{\"num_predict\":24}}" 2>/dev/null) ;;
            chat_batch)
                ms=$(curl -s -m 120 -o /dev/null -w '%{time_total}|%{http_code}' "$BASE/api/chat" \
                    -d "{\"model\":\"$MODEL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"iter $i\"}],\"options\":{\"num_predict\":8}}" 2>/dev/null) ;;
            openai_models)
                ms=$(curl -s -m 30 -o /dev/null -w '%{time_total}|%{http_code}' "$BASE/v1/models" 2>/dev/null) ;;
        esac
        code=${ms##*|}; t=${ms%|*}
        ok=$([ "$code" = "200" ] && echo 1 || echo 0)
        [ "$ok" = "0" ] && fail=$((fail+1))
        echo "$ts,$i,$kind,$code,${t},$ok" >> "$CSV"
    done
    [ $((i % 10)) -eq 0 ] && snap "$(date +%s)"
    sleep 1
done
snap "$(date +%s)"

total=$(tail -n +2 "$CSV" | wc -l)
echo "================ SOAK SUMMARY ================"
echo "requests: $total   failures: $fail   duration: ${MINUTES}m"
echo "p99 ms by kind:"
tail -n +2 "$CSV" | awk -F, '$6==1 {k[$3]=k[$3]" "$5} END {for (x in k) {n=split(k[x],a," "); asort(a); printf "  %-14s p50=%.0f p99=%.0f\n", x, a[int(n/2)*1], a[n]}'}
awk -F, '$6==0 {print "FAILED: iter",$2,$3,"status",$4}' "$CSV" | head -10
[ "$fail" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL ($fail failures)"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
