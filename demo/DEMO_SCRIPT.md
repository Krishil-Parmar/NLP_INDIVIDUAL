# Demo script (draft, about three minutes)

Drafted by Claude Code on 24 Nov 2026 from README section 12.2. Each query's decision below was checked against `demo/app.py` (S2 at its frozen policy v1 threshold) on that date.

The owner rewrites the spoken lines in his own words, rehearses against a timer, and records a backup screen capture. Confidence values appear on screen live, so they are not written here.

## Setup

1. Start the router: `uv run python demo/app.py`. It runs S2 at its frozen threshold from `results/policy_v1.json` (0.485). Wait for "Router ready".
2. Start the Jeff screen harness from `Velm-Main/apps/mobile`: `flutter run -t lib/jeff_router_demo_main.dart -d chrome`.
3. Reset the session counters.

## The six queries

| # | Type (README 12.2) | Type this | Expected decision | Point to make |
| --- | --- | --- | --- | --- |
| 1 | Clear in-scope | `where's my new card, it never arrived` | HANDLE, card_arrival | The matched example shows *why*: a near-paraphrase in the labelled data. No LLM call. |
| 2 | Same request reworded | `my card still hasn't shown up, where is it` | HANDLE, card_arrival | The same decision under rewording. |
| 3 | Between two intents | `is there a charge for this` | HANDLE at 0.485 (a guess between the fee intents) | At this threshold the router guesses. Move the slider to 0.85 and it escalates (step 7). |
| 4 | Clearly unrelated | `what's a good recipe for dinner tonight` | ESCALATE | Low confidence, so the LLM gets it. Policy v1 has no REJECT, so the LLM declines it. |
| 5 | Finance-adjacent, unsupported | `how do i get a mortgage` | HANDLE, failed_transfer (wrong) | The real weakness: at the frozen threshold some out-of-scope queries are handled. Router plus LLM answers 111 of 1,000 out-of-scope test queries, against 24 for the LLM alone (`results/headline_reanalysis.md`, B). |
| 6 | Known failure (error analysis) | `why was my payment declined` | HANDLE, declined_transfer (wrong; Banking77 labels this declined_card_payment) | Very confident and wrong: the nearest labelled example is a near-copy with a different label. This is why S2 covers so little at 1% risk (`results/headline_reanalysis.md`, F). |

## Step 7: the slider

Move the threshold from 0.485 to 0.85. That is about the threshold chosen on validation for a 5% risk bar (0.8457, `results/headline_reanalysis.md`, B). Re-send queries 3 and 5: both now escalate.

The trade-off, on Banking77 test:
- **At 0.485:** about 78% of queries are handled.
- **At 0.85:** about 59% are handled, with fewer errors overall and no more out-of-scope answers than the LLM alone.

Then show the session counters: LLM calls saved and US$ saved.

## Limitations (three sentences, in the owner's words)

1. **It works where a close labelled example exists:** on the assistant's own intents (the Jeff set) it handles almost nothing until it is fine-tuned on their examples (E7, E8).
2. **At the frozen threshold it answers some out-of-scope queries,** so a stricter threshold or a reject rule is needed before real use.
3. **The LLM baseline was weakly prompted** and the Jeff set was written by a Claude model, so both bars are optimistic or circular in places.

## Known mismatch to decide before the day

The screen is Jeff's, but the router behind it uses Banking77's intents. Two options:
- **Say so in one sentence.**
- **Switch the demo to the Jeff intents:** the E8 fine-tuned encoder with the 140 Jeff support examples. This needs a small change to `demo/app.py`.
