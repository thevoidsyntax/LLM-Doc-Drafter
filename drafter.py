#!/usr/bin/env python3
"""
llm-doc-drafter

Converts raw meeting notes into a first-draft set of user stories,
business rules, and open questions.

Output is a draft. It requires review before use.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    print(
        "Missing dependency: anthropic\nInstall with: pip install anthropic",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4000
MAX_INPUT_WORDS = 6000

logger = logging.getLogger("drafter")


class DrafterError(Exception):
    """Base error for expected, handled failures."""


@dataclass
class Config:
    input_path: Path
    output_path: Optional[Path] = None
    output_format: str = "markdown"
    glossary_path: Optional[Path] = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    context: Optional[str] = None
    verbose: bool = False
    dry_run: bool = False
    glossary: Dict[str, str] = field(default_factory=dict)


SYSTEM_PROMPT = """You are assisting a business analyst with a first draft. You are not writing final requirements.

Your job is to read raw meeting notes and extract structure. The notes are messy, incomplete, and may contradict themselves. That is expected.

Rules:

1. Only write requirements that are supported by the notes. If the notes do not specify a threshold, an actor, or an exception path, do not invent one. Put it in the open questions section instead.

2. The open questions section is the most valuable part of your output. Be thorough. Look for: undefined actors, missing thresholds, unstated exception handling, unspecified data retention, absent approval hierarchy, and any place where two statements in the notes conflict.

3. Mark confidence on every user story:
   - HIGH: directly stated in the notes
   - MEDIUM: reasonably inferred, but not stated
   - LOW: guessed to fill a structural gap

4. Do not assign priority. The analyst does that with the sponsor.

5. Preserve domain terms exactly as given in the glossary. Do not translate them.

6. If the notes are too sparse to produce anything useful, say so plainly rather than padding the output."""


USER_TEMPLATE = """Draft user stories from these meeting notes.

{context_block}{glossary_block}
Return your response as JSON with exactly this structure and no other text:

{{
  "summary": "two or three sentences on what the meeting was about",
  "user_stories": [
    {{
      "id": "US-01",
      "actor": "who",
      "story": "As a [actor], I want [goal] so that [reason]",
      "acceptance_criteria": ["criterion", "criterion"],
      "confidence": "HIGH|MEDIUM|LOW",
      "source_note": "the phrase in the notes this came from"
    }}
  ],
  "business_rules": [
    {{
      "id": "BR-01",
      "rule": "statement of the rule",
      "confidence": "HIGH|MEDIUM|LOW",
      "source_note": "supporting phrase"
    }}
  ],
  "open_questions": [
    {{
      "id": "Q-01",
      "question": "the question",
      "why_it_matters": "what breaks if this is not answered",
      "ask": "who should answer this"
    }}
  ],
  "contradictions": [
    {{
      "statement_a": "first statement",
      "statement_b": "conflicting statement",
      "impact": "why this needs resolving"
    }}
  ]
}}

MEETING NOTES:
---
{notes}
---"""


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


def read_notes(path: Path) -> str:
    if not path.exists():
        raise DrafterError(f"Input file not found: {path}")
    if not path.is_file():
        raise DrafterError(f"Input path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("UTF-8 decode failed, retrying with latin-1")
        try:
            text = path.read_text(encoding="latin-1")
        except OSError as exc:
            raise DrafterError(f"Could not read {path}: {exc}") from exc
    except OSError as exc:
        raise DrafterError(f"Could not read {path}: {exc}") from exc

    text = text.strip()
    if not text:
        raise DrafterError(f"Input file is empty: {path}")

    word_count = len(text.split())
    logger.debug("Read %d words from %s", word_count, path)

    if word_count > MAX_INPUT_WORDS:
        raise DrafterError(
            f"Input is {word_count} words, above the {MAX_INPUT_WORDS} limit. "
            "Split the notes and run separately - output quality degrades past this point."
        )
    if word_count < 50:
        logger.warning("Input is only %d words. Output will be thin.", word_count)

    return text


def load_glossary(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise DrafterError(f"Glossary file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DrafterError(f"Glossary is not valid JSON ({path}): {exc}") from exc
    except OSError as exc:
        raise DrafterError(f"Could not read glossary {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DrafterError("Glossary must be a JSON object of term: definition pairs")

    glossary = {str(k): str(v) for k, v in raw.items()}
    logger.debug("Loaded %d glossary terms", len(glossary))
    return glossary


def build_prompt(notes: str, glossary: Dict[str, str], context: Optional[str]) -> str:
    context_block = ""
    if context:
        context_block = f"Domain context: {context}\n\n"

    glossary_block = ""
    if glossary:
        lines = "\n".join(f"- {term}: {definition}" for term, definition in glossary.items())
        glossary_block = (
            "Preserve these domain terms exactly. Do not translate or paraphrase them:\n"
            f"{lines}\n\n"
        )

    return USER_TEMPLATE.format(
        context_block=context_block,
        glossary_block=glossary_block,
        notes=notes,
    )


def get_client() -> "anthropic.Anthropic":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise DrafterError(
            "ANTHROPIC_API_KEY is not set.\n"
            'Set it with: export ANTHROPIC_API_KEY="your-key"'
        )
    return anthropic.Anthropic(api_key=api_key)


def call_model(cfg: Config, prompt: str) -> str:
    client = get_client()
    logger.debug("Model: %s, max_tokens: %d", cfg.model, cfg.max_tokens)

    started = time.monotonic()
    try:
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise DrafterError(f"Authentication failed. Check ANTHROPIC_API_KEY. ({exc})") from exc
    except anthropic.RateLimitError as exc:
        raise DrafterError(f"Rate limited. Wait and retry. ({exc})") from exc
    except anthropic.APIStatusError as exc:
        raise DrafterError(f"API returned {exc.status_code}: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise DrafterError(f"Could not reach the API. Check network. ({exc})") from exc

    elapsed = time.monotonic() - started
    logger.debug("Response in %.1fs", elapsed)

    if hasattr(response, "usage"):
        logger.debug(
            "Tokens in: %s, out: %s",
            getattr(response.usage, "input_tokens", "?"),
            getattr(response.usage, "output_tokens", "?"),
        )

    text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    if not text_parts:
        raise DrafterError("Model returned no text content")

    return "\n".join(text_parts)


def parse_response(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:300]
        raise DrafterError(
            f"Model did not return valid JSON: {exc}\n\nFirst 300 chars:\n{preview}"
        ) from exc

    if not isinstance(data, dict):
        raise DrafterError("Model returned JSON, but not an object")

    for key in ("user_stories", "business_rules", "open_questions", "contradictions"):
        data.setdefault(key, [])
        if not isinstance(data[key], list):
            logger.warning("Field '%s' was not a list, coercing to empty", key)
            data[key] = []

    data.setdefault("summary", "")
    return data


def render_markdown(data: Dict[str, Any], source: Path) -> str:
    out: List[str] = []
    out.append("# Draft User Stories")
    out.append("")
    out.append(f"Source: `{source.name}`  ")
    out.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    out.append("")
    out.append("> Draft output. Every item needs review against the original notes before use.")
    out.append("")

    if data.get("summary"):
        out.append("## Summary")
        out.append("")
        out.append(str(data["summary"]))
        out.append("")

    stories = data.get("user_stories", [])
    out.append(f"## User Stories ({len(stories)})")
    out.append("")
    if not stories:
        out.append("None extracted.")
        out.append("")
    for s in stories:
        out.append(f"### {s.get('id', 'US-??')} - {s.get('actor', 'unspecified actor')}")
        out.append("")
        out.append(f"**Confidence:** `{s.get('confidence', 'UNKNOWN')}`")
        out.append("")
        out.append(str(s.get("story", "")))
        out.append("")
        criteria = s.get("acceptance_criteria", [])
        if criteria:
            out.append("**Acceptance criteria**")
            out.append("")
            for c in criteria:
                out.append(f"- {c}")
            out.append("")
        if s.get("source_note"):
            out.append(f"*From notes:* {s['source_note']}")
            out.append("")

    rules = data.get("business_rules", [])
    out.append(f"## Business Rules ({len(rules)})")
    out.append("")
    if rules:
        out.append("| ID | Rule | Confidence | Source |")
        out.append("|---|---|---|---|")
        for r in rules:
            rule = str(r.get("rule", "")).replace("|", "\\|")
            src = str(r.get("source_note", "")).replace("|", "\\|")
            out.append(
                f"| {r.get('id', '')} | {rule} | `{r.get('confidence', '')}` | {src} |"
            )
    else:
        out.append("None extracted.")
    out.append("")

    questions = data.get("open_questions", [])
    out.append(f"## Open Questions ({len(questions)})")
    out.append("")
    out.append("This is the section worth reading first.")
    out.append("")
    if questions:
        for q in questions:
            out.append(f"**{q.get('id', 'Q-??')}** - {q.get('question', '')}")
            out.append("")
            if q.get("why_it_matters"):
                out.append(f"- Matters because: {q['why_it_matters']}")
            if q.get("ask"):
                out.append(f"- Ask: {q['ask']}")
            out.append("")
    else:
        out.append("None identified. That usually means the notes were too sparse, not that everything is clear.")
        out.append("")

    contradictions = data.get("contradictions", [])
    if contradictions:
        out.append(f"## Contradictions ({len(contradictions)})")
        out.append("")
        for c in contradictions:
            out.append(f"- **A:** {c.get('statement_a', '')}")
            out.append(f"- **B:** {c.get('statement_b', '')}")
            out.append(f"- **Impact:** {c.get('impact', '')}")
            out.append("")

    return "\n".join(out)


def write_output(content: str, path: Optional[Path]) -> None:
    if path is None:
        print(content)
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise DrafterError(f"Could not write to {path}: {exc}") from exc
    print(f"Written to {path}", file=sys.stderr)


def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="drafter",
        description="Draft user stories from meeting notes. Output requires review.",
    )
    parser.add_argument("--input", "-i", required=True, help="Path to notes file")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    parser.add_argument(
        "--format", "-f", choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument("--glossary", "-g", default=None, help="JSON file of domain terms")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max response tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument("--context", default=None, help='Domain context, e.g. "retail lending"')
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt and exit")

    args = parser.parse_args(argv)

    if args.max_tokens < 500:
        parser.error("--max-tokens below 500 will truncate the output")

    return Config(
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        output_format=args.format,
        glossary_path=Path(args.glossary) if args.glossary else None,
        model=args.model,
        max_tokens=args.max_tokens,
        context=args.context,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def run(cfg: Config) -> int:
    notes = read_notes(cfg.input_path)
    cfg.glossary = load_glossary(cfg.glossary_path)
    prompt = build_prompt(notes, cfg.glossary, cfg.context)

    if cfg.dry_run:
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT)
        print()
        print("=== USER ===")
        print(prompt)
        print()
        word_total = len(SYSTEM_PROMPT.split()) + len(prompt.split())
        print(f"=== Approx {word_total} words ===", file=sys.stderr)
        return 0

    raw = call_model(cfg, prompt)
    data = parse_response(raw)

    if cfg.output_format == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        content = render_markdown(data, cfg.input_path)

    write_output(content, cfg.output_path)

    low = sum(1 for s in data.get("user_stories", []) if s.get("confidence") == "LOW")
    if low:
        print(f"Note: {low} story/stories marked LOW confidence. Check those first.", file=sys.stderr)

    return 0


def main() -> int:
    cfg = parse_args()
    setup_logging(cfg.verbose)
    try:
        return run(cfg)
    except DrafterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
