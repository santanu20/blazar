# blazar-core

Pure domain core of the [blazar](https://github.com/santanu20/blazar) local
inference platform: configuration, SQLite-backed store, model catalog, GGUF
metadata parsing, and the profile compiler that turns model shape plus policy
into an engine command line. No I/O, no async runtime — everything here is
testable domain logic.

Part of the `blazar` workspace; see the
[main README](https://github.com/santanu20/blazar) for the full platform.
License: MIT OR Apache-2.0
