# Code-Switching TTS Benchmark

`manifest.jsonl` is the 1,200-utterance intra-sentential code-switching corpus
introduced in the paper. It is a **mixed-language text corpus**: it contains no
audio. The speech under evaluation is always synthesized by the systems being
compared, from the transcripts here plus a fixed reference voice prompt.

## Composition

| | |
|---|---|
| Utterances | 1,200 |
| Directions | 12 (100 utterances each) |
| Languages | English, German, French, Japanese, Korean |
| Embedded phrases | 4,266 total, 3.56 per utterance (range 3–5) |
| Phrase length | 5.72 words; 19.8 chars for Japanese/Korean, 49.6 for Latin-script |
| Utterance length | 363 characters on average (210–549) |
| Topics | 18 domains |
| Registers | 6 styles |

The corpus deliberately targets **dense multi-word phrasal insertions**, the
regime where cross-lingual accent leakage is most pronounced, rather than the
single-word alternations that dominate prior code-switching benchmarks.

**Directions** (matrix carrier → embedded phrase language):

| Matrix | Embedded | Direction codes |
|---|---|---|
| English  | Japanese, Korean       | `E_to_J`, `E_to_K` |
| Japanese | English, German, French| `J_to_E`, `JA_to_DE`, `JA_to_FR` |
| Korean   | English, German, French| `K_to_E`, `KO_to_DE`, `KO_to_FR` |
| German   | Japanese, Korean       | `DE_to_JA`, `DE_to_KO` |
| French   | Japanese, Korean       | `FR_to_JA`, `FR_to_KO` |

**Topics**: academic, aesthetics, anime_manga, business, culture, daily_life,
entertainment, etiquette, family_kinship, finance, food, lifestyle_fashion,
literary, marketing, philosophy, seasonal_nature, tech, travel

**Registers**: casual, contemplative, formal, literary, narrative, review

## Schema

One JSON object per line:

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique utterance id, e.g. `syncs_E_to_K_0` |
| `dataset` | str | Always `synthetic_cs` |
| `setup` | str | Direction code, e.g. `E_to_K` |
| `base_lang` | str | Matrix carrier language (ISO 639-1) |
| `phrase_lang` | str | Embedded phrase language (ISO 639-1) |
| `text` | str | The code-switched utterance to synthesize |
| `text_phr` | str | Monolingual rendering, with every embedded phrase replaced by its translation. Provided for reference; **not used by the paper's pipeline**, which feeds the same `text` to both conditional branches and varies only the language tag |
| `phrases` | list[str] | Embedded phrases, in order of appearance |
| `phrases_translated` | list[str] | Matrix-language gloss of each phrase |
| `phrase_char_spans` | list[[int,int]] | Character spans of each phrase in `text`, end-exclusive. `text[s:e] == phrases[i]` holds for every entry |
| `language` | str | Utterance-level language tag passed to the backbone (equals `base_lang`) |
| `ref_audio_path` | str | Reference voice prompt, relative to the repository root |
| `ref_text` | str | Transcript of the reference prompt |
| `meta.topic` | str | One of the 18 domains |
| `meta.style` | str | One of the 6 registers |
| `meta.n_phrases` | int | Number of embedded phrases |

`phrase_char_spans` is what LCG consumes to locate the embedded region; the
spans are byte-exact against `text` for all 1,200 entries.

## Quick look

```bash
# first entry, pretty-printed
head -1 benchmark/manifest.jsonl | python -m json.tool

# utterances per direction
python -c "import json,collections;print(collections.Counter(json.loads(l)['setup'] for l in open('benchmark/manifest.jsonl')))"
```

## Example

```json
{
  "id": "syncs_E_to_K_0",
  "setup": "E_to_K",
  "base_lang": "en",
  "phrase_lang": "ko",
  "text": "I read the essay collection beside the river, and its best pages felt like 은은한 달빛이 솔잎 끝에 맺힌 밤, restrained but luminous. ...",
  "phrases": ["은은한 달빛이 솔잎 끝에 맺힌 밤", "..."],
  "phrases_translated": ["a night when soft moonlight gathers on pine needles", "..."],
  "phrase_char_spans": [[75, 93], [187, 210], [240, 260], [329, 345]],
  "ref_audio_path": "examples/en_ref.wav",
  "meta": {"topic": "literary", "style": "review", "n_phrases": 4}
}
```

## Construction

The corpus was written by an LLM pipeline (`gpt-5.5-2026-04-23`, xhigh reasoning
effort) prompted to produce sentences a bilingual speaker might plausibly say,
then filtered so that every embedded phrase is idiomatic in its own language,
carries real cultural or technical grounding, and sits in a grammatical slot of
the matrix sentence. Phrase spans were recomputed from the final text and
verified programmatically. Coverage across directions, topics, registers, and
phrase counts is balanced by construction.

The generation scripts are not released: the pipeline changed during
construction, so the shipped corpus is the artifact of record. Full
methodological detail is in Section 4.1 of the paper.

## Usage

```bash
python synth/synth.py --manifest benchmark/manifest.jsonl ...
```

Reference prompts referenced by `ref_audio_path` live in `examples/` and follow
the terms of their source corpora; see `examples/README.md`.
