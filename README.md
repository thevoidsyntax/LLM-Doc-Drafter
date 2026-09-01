# llm-doc-drafter

Takes raw meeting notes and produces a first-draft set of user stories and open questions.

## What this is not

This does not write requirements. It produces a starting draft that a BA edits. The output is wrong often enough that shipping it unreviewed would be worse than writing from scratch.

The value is in the open-questions section. The model is reliably good at spotting what the notes did not say — missing actors, undefined thresholds, unstated exception handling. It is unreliable at deciding what the requirement should be.

## Known limitations

| Limitation | Detail |
|---|---|
| Invents acceptance criteria | If the notes are vague, it fills gaps with plausible-sounding rules that nobody agreed to. Check every criterion against the notes. |
| Loses Indonesian business terms | Terms like *nota dinas* or *disposisi* get translated into generic English equivalents that lose meaning. Add them to the glossary file. |
| No sense of priority | Everything comes out as "Must". Reprioritise manually. |
| Long notes degrade output | Above roughly 4000 words the later sections get thin. Split the input. |

## Requirements

```
python 3.9+
anthropic>=0.39.0
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```bash
# Build the prompt without calling the API
python drafter.py --input examples/example-meeting.txt --dry-run

# Basic run
python drafter.py --input examples/example-meeting.txt

# With glossary, written to a file
python drafter.py --input examples/example-meeting.txt --glossary glossary.json --output draft.md

# JSON output for tooling
python drafter.py --input examples/example-meeting.txt --format json --output draft.json
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | required | Path to notes file |
| `--output`, `-o` | stdout | Output file path |
| `--format`, `-f` | markdown | `markdown` or `json` |
| `--glossary`, `-g` | none | JSON file of domain terms to preserve |
| `--model` | claude-sonnet-4-6 | Model identifier |
| `--max-tokens` | 4000 | Response limit |
| `--context` | none | Extra context, e.g. "retail lending" |
| `--verbose`, `-v` | off | Log prompt size and timing |
| `--dry-run` | off | Build the prompt and exit without calling the API |

## Glossary format

```json
{
  "nota dinas": "internal memo, formal, requires signature",
  "disposisi": "routing instruction written by a superior on an incoming document"
}
```

| LLM API integration | Written and dry-run tested. Not yet run against the live API. |

Terms in the glossary are passed to the model with instructions to keep the original term and treat the definition as context.

## Cost

A 2000-word note costs roughly a few cents per run. Use `--dry-run` while adjusting the prompt.
