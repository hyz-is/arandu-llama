# Chat templates

The first three files are the `tokenizer.chat_template` value of a published
GGUF, copied byte for byte from the file's metadata; all three are Apache-2.0.
`ornith-1.5-9b.jinja` is the student's `chat_template.jinja`, downloaded byte
for byte from the model repository at the pinned revision, and is MIT.

| file | source | revision | licence | sha256 |
| --- | --- | --- | --- | --- |
| `gemma-4-12b-it.jinja` | `google/gemma-4-12B-it-qat-q4_0-gguf`, `gemma-4-12b-it-qat-q4_0.gguf` | `29d097773436b69ff9feafd636ab4cf873786537` | apache-2.0 | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` |
| `gpt-oss-20b.jinja` | `ggml-org/gpt-oss-20b-GGUF`, `gpt-oss-20b-MXFP4.gguf` | `ef9b12f2ff56c69cf32153a02784e7a3c88bf524` | apache-2.0 | `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146` |
| `qwen3.8-27b.jinja` | `ggml-org/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf` | `71bc7b627595dc8a91039addd9c791ae548d6747` | apache-2.0 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `ornith-1.5-9b.jinja` | `ornith-ai/Ornith-1.5-9B`, `chat_template.jinja` | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` | mit | `9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235` |

The Ornith file is rebuilt with:

```sh
curl -sSL -o ornith-1.5-9b.jinja \
  https://huggingface.co/ornith-ai/Ornith-1.5-9B/resolve/489cb97981b8654bcfcf30ce1f94ed1b62e07b53/chat_template.jinja
```
