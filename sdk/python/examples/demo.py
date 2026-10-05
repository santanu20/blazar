"""End-to-end demo against a live Blazar gateway.

    PYTHONPATH=sdk/python python3 examples/demo.py [model]

Prints a chat reply, a streaming reply, and the operational cards.
Exits non-zero with the teaching message when the gateway refuses.
"""

import sys

from blazar import BlazarError, Client


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b:q4_0"
    blazar = Client()

    reply = blazar.chat(
        model,
        [{"role": "user", "content": "Say 'demo ok' and nothing else."}],
        max_tokens=256,
    )
    text = reply["choices"][0]["message"]["content"]
    print("chat:", text.strip()[:120])

    print("stream:", end=" ")
    for chunk in blazar.chat(
        model,
        [{"role": "user", "content": "Count: 1 2 3"}],
        stream=True,
        max_tokens=256,
    ):
        choices = chunk.get("choices") or []  # role/usage frames carry none
        if not choices:
            continue
        delta = choices[0].get("delta", {}).get("content", "")
        print(delta, end="", flush=True)
    print()

    residents = blazar.ps()["models"]
    for row in residents:
        sleeping = row.get("blazar_sleeping")
        state = " (sleeping)" if sleeping is True else ""
        print(f"ps: {row['name']} via {row['blazar_engine']}{state}")

    card = blazar.explain(model)
    print("explain: lane", card.get("lane", card.get("engine", "?")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BlazarError as e:
        print(f"gateway said: {e.message}", file=sys.stderr)
        raise SystemExit(2) from None
