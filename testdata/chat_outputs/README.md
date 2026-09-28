# Chat outputs

Each file is one generation from a teacher smoke run, written by `jq` from the
run's report and never edited by hand; `output` decodes to the report's value
byte for byte. `output` is `readings[index].content` of the run's report:
the run used `ReasoningFormatNone`, so that is the model's whole text, special
tokens written out, before any parse. `user` is the one user turn the run sent,
`question + "\n\n" + instruction`, with the question from the dataset's smoke
split. `template` names the file under `../chat_templates/` whose digest the
report records, and `finish` is how the generation stopped.

The run was `ChatOptions{MaxTokens: 1024, Temperature: 0, TopK: 1, Seed: 7}`,
`EnableThinking` unset (on), engine `arandu-llama.chat-jinja.v1/llama.cpp.90c26fcd`.

| file | report | report sha256 | index | id | finish |
| --- | --- | --- | --- | --- | --- |
| `gpt-oss-20b.gsm8k-train-1468.json` | `gpt-oss-20b-mxfp4.json` | `03a44912f2a8c2ba2b91d5bd28c5c010d1d92752e963653247a71e0e22c3dd55` | 0 | `gsm8k-train-1468` | stop |
| `qwen3.8-27b.gsm8k-train-1468.json` | `qwen3-8-27b-q4-k-m.json` | `984f2ba2909534fcf68fbc6c88e0a2f24863c4c969b23e4a1adf9af9f141347f` | 0 | `gsm8k-train-1468` | stop |
| `gemma-4-12b-it.gsm8k-train-6546.json` | `gemma-4-12b-it-qat-q4-0.json` | `2954695ead40f8d28b6baf457a0761fd96df758d2c0126c1d905d71aee05718d` | 4 | `gsm8k-train-6546` | stop |
| `gemma-4-12b-it.gsm8k-train-1468.json` | `gemma-4-12b-it-qat-q4-0.json` | `2954695ead40f8d28b6baf457a0761fd96df758d2c0126c1d905d71aee05718d` | 0 | `gsm8k-train-1468` | length |

The reports are `runtime/tcap-teachers-20260927/smoke/jinja/` in the tayi-ai
repository. The questions are `30-training/datasets/tcap-math-v1/smoke.jsonl`
there, sha256 `1872db6bf59030c0db78d6b8d4eb1b23c3cd8d2e3ac82b35d400792b3f5fba33`.

A file is rebuilt with, for example:

```sh
jq -n --arg template gpt-oss-20b --arg id gsm8k-train-1468 \
  --argjson question "$(jq -c 'select(.id=="gsm8k-train-1468") | .question' smoke.jsonl)" \
  --arg instruction "Solve the problem. Write the final numeric answer on the last line as '#### <number>'." \
  --arg finish stop --slurpfile src gpt-oss-20b-mxfp4.json --argjson idx 0 \
  '{template: $template, id: $id, user: ($question + "\n\n" + $instruction), finish: $finish, output: $src[0].readings[$idx].content}'
```

`finish` is read off the report, which has no finish reason: `output_tokens`
there counts the text tokenized again, BOS included. 1025 is the 1024-token
ceiling plus BOS, so that output was cut; the others end well below it.

Licences are per artefact. The three models are Apache-2.0, as their
templates are. The GSM8K questions in `user` are MIT
(`openai/grade-school-math`).
