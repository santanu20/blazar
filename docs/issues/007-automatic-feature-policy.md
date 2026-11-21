Title: Centralized automatic feature policy

Priority: P0 | Wave: 2 | Backlog: B03 | Effort: M | Depends: -

## Problem

Automatic behavior is real and shipped, but scattered across independent
knobs (all documented in docs/7.SETUP.md): slots=0 auto sizing, the
cache_type auto KV-quantization ladder, spec=auto family selection with
n-gram fallback (spec_auto_ngram) and the adaptive speculation governor
(spec_auto_manage), adaptive_slots, adaptive_decode_gate, cache_idle_slots,
and ubatch_auto (opt-in). Precedence (explicit user pin wins) is enforced
implicitly at each site instead of declared once.

## Approach

1. FeaturePolicy per managed feature: Disabled | SafeAuto | MeasuredAuto |
   Experimental | UserPinned.
2. One policy table as the single source of truth, with precedence enforced
   in one place: explicit user config > verified model profile > measured
   automatic policy > safe default. This prevents automatic management from
   repeatedly undoing deliberate operator choices.
3. blazar explain shows each feature's policy and decision.
4. Zero default-behavior change in this issue: pin current behavior with
   tests first, then migrate enforcement sites to the policy table.

## Acceptance criteria

- [ ] Every automatic feature has an explicit policy entry
- [ ] Precedence enforced centrally with unit tests
- [ ] User-pinned settings never rewritten (regression test)
- [ ] blazar explain exposes per-feature policy decisions
- [ ] Behavior-preservation tests prove no default change
