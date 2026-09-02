# Reference Voice Prompts

These five monolingual prompts are the matrix-language speakers used throughout
the benchmark. Every directional task is synthesized from a single fixed prompt
so that speaker identity is held constant and accent leakage can be isolated
from identity artifacts.

Each file is a short excerpt taken from a publicly available monolingual TTS
corpus. **Terms of use follow the policy of the source project** — please check
the corresponding repository before redistributing or reusing these clips.

| File | Language | Source corpus | Used as matrix carrier for |
|------|----------|---------------|-----------------------------|
| `en_ref.wav` | English  | [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) | `E_to_J`, `E_to_K` |
| `ja_ref.wav` | Japanese | [CSS10](https://github.com/Kyubyong/css10) | `J_to_E`, `JA_to_DE`, `JA_to_FR` |
| `ko_ref.wav` | Korean   | [KSS Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset) | `K_to_E`, `KO_to_DE`, `KO_to_FR` |
| `de_ref.wav` | German   | [CML-TTS](https://www.openslr.org/146/) | `DE_to_JA`, `DE_to_KO` |
| `fr_ref.wav` | French   | [CML-TTS](https://www.openslr.org/146/) | `FR_to_JA`, `FR_to_KO` |

> **Note on `ko_ref.wav`.** The KSS Dataset is distributed under a
> **non-commercial** license. This clip, and any audio cloned from it, may not
> be used commercially. All work in this repository is non-commercial research.

All files are mono WAV clips, kept byte-identical to the excerpts used for the
paper (formats vary by source corpus: 22.05–44.1 kHz, PCM-16 or float; the
synthesis pipeline resamples internally). The paths above are referenced
directly by `ref_audio_path` in `benchmark/manifest.jsonl`, so this directory
should not be renamed.

## Transcripts

The transcript accompanying each prompt is stored in the `ref_text` field of the
manifest and is supplied to the model alongside the audio for zero-shot cloning.

### `en_ref.wav` — English (LJSpeech)
> with the particular purposes of the agency involved. The Commission recognizes that this is a controversial area

### `ja_ref.wav` — Japanese (CSS10)
> この前探った時は、途中に瘢痕の隆起があったので、ついそこが行きどまりだとばかり思って、ああ云ったんですが、

### `ko_ref.wav` — Korean (KSS)
> 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다.

### `de_ref.wav` — German (CML-TTS)
> aber meinen weißen Brüdern darf ich es anvertrauen, wenn sie mir versprechen, demselben nicht nachzuspüren

### `fr_ref.wav` — French (CML-TTS)
> puis dans une salle de tapis où brûle une haute lampe d'or, elles se couchent au hasard

## Citing the source corpora

If you use these prompts in derived work, please cite the original corpora.

```bibtex
@misc{ljspeech17,
  author       = {Keith Ito and Linda Johnson},
  title        = {The {LJ Speech Dataset}},
  year         = 2017,
  howpublished = {\url{https://keithito.com/LJ-Speech-Dataset/}},
}

@inproceedings{park2019css10,
  author    = {Kyubyong Park and Thomas Mulc},
  title     = {{CSS10}: A Collection of Single Speaker Speech Datasets for 10 Languages},
  booktitle = {Interspeech},
  year      = 2019,
}

@misc{park2018kss,
  author       = {Kyubyong Park},
  title        = {{KSS} Dataset: Korean Single Speaker Speech Dataset},
  year         = 2018,
  howpublished = {\url{https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset}},
}

@inproceedings{oliveira2023cmltts,
  author    = {Frederico S. Oliveira and Edresson Casanova and Arnaldo Candido Junior and
               Anderson S. Soares and Arlindo R. Galv{\'a}o Filho},
  title     = {{CML-TTS}: A Multilingual Dataset for Speech Synthesis in Low-Resource Languages},
  booktitle = {Text, Speech, and Dialogue (TSD)},
  pages     = {188--199},
  publisher = {Springer},
  year      = 2023,
}
```
