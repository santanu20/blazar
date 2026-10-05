"""Realtime voice + metadata + Anthropic batch, one script.

Run against a live daemon:

    BLAZAR_URL=http://127.0.0.1:11501 PYTHONPATH=sdk/python \
        python3 sdk/python/examples/realtime_voice.py qwen3-0.6b:q4_0

The first utterance is synthesized with the gateway's own TTS lane
(Piper), so the script needs a pulled voice
(`blazar tts --pull en_US-amy-medium`) and a whisper model
(`blazar whisper --pull tiny`).
"""

import json
import sys
import time

from blazar import BlazarError, Client


def speech_pcm(blazar: Client, text: str) -> bytes:
    """Make real speech via the TTS lane; strip the 44-byte RIFF header.

    Piper emits PCM16 mono 22.05 kHz; the realtime lane wraps it in a WAV
    header for whisper, which resamples on decode.
    """
    import urllib.request

    req = urllib.request.Request(
        blazar.base_url + "/v1/audio/speech",
        data=json.dumps(
            {"model": "en_US-amy-medium", "input": text, "response_format": "wav"}
        ).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {blazar.api_key}"} if blazar.api_key else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=blazar.timeout) as resp:
        wav = resp.read()
    assert wav[:4] == b"RIFF", f"not a wav: {wav[:12]!r}"
    return wav[44:]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b:q4_0"
    blazar = Client()

    # -- metadata cards ------------------------------------------------------
    reply = blazar.chat(
        model,
        [{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=64,
        metadata={"trace": "demo-1"},
    )
    card = blazar.metadata_update(reply["id"], {"trace": "demo-2"})
    print(f"metadata card: id={card['id']} metadata={card['metadata']}")

    # -- realtime voice --------------------------------------------------------
    pcm = speech_pcm(blazar, "Hello. The weather in Tokyo today is the topic.")
    with blazar.realtime(model) as rt:
        out = rt.voice_turn(pcm)
    print(f"transcript: {out['transcript'].strip()!r}")
    print(f"reply:      {out['reply'].strip()!r}")
    print(f"audio:      {len(out['audio'])} PCM bytes")

    # -- anthropic batch -------------------------------------------------------
    batch = blazar.anthropic_batch_create(
        [
            {
                "custom_id": f"demo-{i}",
                "params": {
                    "model": model,
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": f"count to {i + 1}"}],
                },
            }
            for i in range(2)
        ]
    )
    print(f"batch: {batch['id']} {batch['processing_status']}")
    while True:
        state = blazar.anthropic_batch(batch["id"])
        if state["processing_status"] == "ended":
            break
        time.sleep(1)
    counts = state["request_counts"]
    print(f"batch ended: succeeded={counts['succeeded']} errored={counts['errored']}")
    for row in blazar.anthropic_batch_results(batch["id"]):
        print(f"  row {row['custom_id']}: {row['result']['type']}")


if __name__ == "__main__":
    try:
        main()
    except BlazarError as e:
        print(f"blazar error: {e}", file=sys.stderr)
        sys.exit(1)
