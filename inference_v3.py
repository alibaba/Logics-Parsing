"""Run Logics-Parsing-V3 on a PDF, an image folder, or one image.

A one-page input uses the single-page prompt; longer inputs are inferred in
non-overlapping windows with state passed to the next window. The model output is saved
as raw DSL, Markdown, and a Markdown outline.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fitz
from PIL import Image
from transformers import AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
WINDOW_SIZE = 2
MIN_PIXELS = 65536
MAX_PIXELS = 2048 * 2048
RESIZE_FACTOR = 32
PDF_RENDER_DPI = 200
MAX_NEW_TOKENS = 16384
POSTPROCESS_VERSION = "adjacent_paired_continuation_v2_no_trim"
STATE_MAX_CLOSED = 32
STATE_TAIL_CHARS = 200

SYSTEM_MESSAGE = "You are a helpful assistant"

LONG_DOCUMENT_PROMPT = """Task mode: longdoc_parse.
The attached page inputs are consecutive pages from a long document. Parse only these pages and output structured document labels.
Use the previous-window state below only as context for continuing the hierarchy or unfinished content. Do not copy the state into the answer.

Output rules:
- Output labels only, one item per line. Do not output markdown, JSON, explanations, or extra text.
- Follow the natural reading order across the current pages.
- Every label includes an absolute 0-based page id. Figure, chart and table locations, plus caption references to them, also include bounding-box coordinates normalized from 0 to 1000.
- Title: <|title|><|level|>N<|text|>TITLE_TEXT<|loc|>page, where N is 1-7.
- Text: <|text|>TEXT<|loc|>page.
- Table: <|table|>TABLE_HTML_OR_TEXT<|loc|>page,x1,y1,x2,y2.
- Figure/chart/formula: <|figure|><|loc|>page,x1,y1,x2,y2, <|chart|><|loc|>page,x1,y1,x2,y2, or <|formula|>FORMULA_TEXT<|loc|>page.
- Caption: <|caption|>CAPTION_TEXT<|loc|>page<|ref|>target_page,x1,y1,x2,y2.
- Use <|truncated|> when a text/table block continues later, and <|continued|> when it continues from earlier pages.
- For cross-column or merged text/table blocks, insert <break> at each content discontinuity and reduce each <|loc|> marker to its page id.
- Finish after the final complete label. Do not add a separate terminator line to the DSL.

Window metadata:
- Window index: {window_idx} (0-based)
- Window number: {window_number} / {total_windows}
- Total document pages: {total_pages}
- Is final window: {is_final_window}
Current page ids: {pages}

Previous-window state:
{state}
"""

SINGLE_PAGE_PROMPT = """Task mode: single_page_parse.
The attached page input is one page from a document. Parse only this page and output structured document labels.

Output rules:
- Output labels only, one item per line. Do not output markdown, JSON, explanations, or extra text.
- Follow the natural reading order on the current page.
- Only figure, chart and table labels carry a location: <|loc|>page,x1,y1,x2,y2, using the 0-based page id shown below and coordinates normalized as integers from 0 to 1000. All other labels carry no location.
- Title: <|title|>TITLE_TEXT.
- Text: <|text|>TEXT.
- Table: <|table|>TABLE_HTML_OR_TEXT<|loc|>page,x1,y1,x2,y2.
- Figure/chart: <|figure|><|loc|>page,x1,y1,x2,y2, or <|chart|><|loc|>page,x1,y1,x2,y2.
- Formula: <|formula|>FORMULA_TEXT.
- Caption: <|caption|>CAPTION_TEXT.
- Do not output title levels, hierarchy open/close tokens, continuation/truncation markers, document state, or markdown.
- Finish after the final complete label. Do not add a separate terminator line to the DSL.

Current page id: 0
"""

TITLE_STATE_RE = re.compile(
    r"<\|title\|><\|level\|>(?P<level>\d+)<\|text\|>(?P<text>.*?)(?=<\|loc\|>|$)"
)
# Keep page-only title locations, matching the training state.
STATE_LOC_RE = re.compile(r"<\|loc\|>(?P<loc>\d+(?:,\d+,\d+,\d+,\d+)?)")
DSL_TAG_RE = re.compile(r"<\|[^>]+?\|>")


@dataclass
class OpenTitle:
    level: int
    text: str
    loc: str = ""

    def to_state_line(self) -> str:
        line = f"<|title|><|level|>{self.level}<|text|>{self.text}"
        return line + (f"<|loc|>{self.loc}" if self.loc else "")


@dataclass
class WindowState:
    closed_sections: list[str] = field(default_factory=list)
    open_titles: list[OpenTitle] = field(default_factory=list)
    tail: Optional[str] = None

    def format(self) -> str:
        lines = ["<|state|>"]
        for section in self.closed_sections[-STATE_MAX_CLOSED:]:
            lines.append(f"<|section|>{section}<|closed|>")
        if self.open_titles:
            lines.append("<|open_tree|>")
            lines.extend(title.to_state_line() for title in self.open_titles)
            lines.append("<|/open_tree|>")
        if self.tail:
            lines.append(f"<|tail|>{self.tail}<|truncated|>")
        lines.append("<|/state|>")
        return "\n".join(lines)

    def update(self, output: str) -> None:
        last_content = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            title_match = TITLE_STATE_RE.search(line)
            if title_match:
                loc_match = STATE_LOC_RE.search(line)
                self._push_title(
                    OpenTitle(
                        level=int(title_match.group("level")),
                        text=" ".join(title_match.group("text").split()),
                        loc=loc_match.group("loc") if loc_match else "",
                    )
                )
            if line.startswith(
                (
                    "<|text|>",
                    "<|table|>",
                    "<|image|>",
                    "<|figure|>",
                    "<|chart|>",
                    "<|formula|>",
                    "<|caption|>",
                    "<|continued|>",
                )
            ):
                text = " ".join(DSL_TAG_RE.sub(" ", line).split())
                last_content = (
                    "..." + text[-STATE_TAIL_CHARS:]
                    if len(text) > STATE_TAIL_CHARS
                    else text
                )
                if "<|truncated|>" in line:
                    self.tail = last_content
        if last_content and not output.rstrip().endswith("<|truncated|>"):
            self.tail = last_content

    def _push_title(self, title: OpenTitle) -> None:
        while self.open_titles and self.open_titles[-1].level >= title.level:
            closed = self.open_titles.pop()
            if closed.text:
                self.closed_sections.append(closed.text)
        self.open_titles.append(title)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    if height < RESIZE_FACTOR or width < RESIZE_FACTOR:
        raise ValueError(f"Image is too small: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Image aspect ratio is too large: {width}x{height}")

    target_h = round(height / RESIZE_FACTOR) * RESIZE_FACTOR
    target_w = round(width / RESIZE_FACTOR) * RESIZE_FACTOR
    if target_h * target_w > MAX_PIXELS:
        scale = math.sqrt((height * width) / MAX_PIXELS)
        target_h = math.floor(height / scale / RESIZE_FACTOR) * RESIZE_FACTOR
        target_w = math.floor(width / scale / RESIZE_FACTOR) * RESIZE_FACTOR
    elif target_h * target_w < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (height * width))
        target_h = math.ceil(height * scale / RESIZE_FACTOR) * RESIZE_FACTOR
        target_w = math.ceil(width * scale / RESIZE_FACTOR) * RESIZE_FACTOR
    return target_h, target_w


def save_model_image(image: Image.Image, destination: Path) -> None:
    image = image.convert("RGB")
    width, height = image.size
    target_h, target_w = smart_resize(height, width)
    if (target_w, target_h) != image.size:
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    image.save(destination, format="PNG")


def image_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]), path.name) if numbers else (10**12, path.name)


def prepare_input(input_path: Path, work_dir: Path) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        sources = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file()
                and not path.name.startswith((".", "._"))
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=image_sort_key,
        )
        if not sources:
            raise ValueError(f"No supported images found in {input_path}")
        pages = []
        for page_index, source in enumerate(sources):
            destination = work_dir / f"page_{page_index:04d}.png"
            with Image.open(source) as image:
                save_model_image(image, destination)
            pages.append(destination)
        return pages

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if input_path.suffix.lower() == ".pdf":
        pages = []
        with fitz.open(input_path) as document:
            page_count = len(document)
            if not page_count:
                raise ValueError(f"PDF contains no pages: {input_path}")
            for page_index in range(page_count):
                page = document[page_index]
                scale = PDF_RENDER_DPI / 72
                raster = (page.rect * fitz.Matrix(scale, scale)).irect
                height, width = smart_resize(raster.height, raster.width)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(width / page.rect.width, height / page.rect.height),
                    alpha=False,
                )
                destination = work_dir / f"page_{page_index:04d}.png"
                if (pixmap.width, pixmap.height) == (width, height):
                    pixmap.save(destination)
                else:
                    with Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples) as image:
                        resized = image.resize((width, height), Image.Resampling.LANCZOS)
                        resized.save(destination, format="PNG")
                        resized.close()
                pages.append(destination)
        return pages

    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported input: {input_path}")
    destination = work_dir / "page_0000.png"
    with Image.open(input_path) as image:
        save_model_image(image, destination)
    return [destination]


def sanitize_output(output: str) -> str:
    lines = []
    for raw_line in output.splitlines():
        line = re.sub(r"<?\|end\|>", "", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


class DocumentParser:
    def __init__(
        self, model_path: str, window_size: int = WINDOW_SIZE,
        max_model_len: int = 32768, gpu_memory_utilization: float = 0.8,
        enforce_eager: bool = False,
    ):
        import vllm
        from vllm import LLM, SamplingParams

        if vllm.__version__ != "0.22.1":
            raise RuntimeError(f"Expected vLLM 0.22.1, got {vllm.__version__}")

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.stats = []
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Preserve checkpoint EOS plus the tokenizer's chat-turn EOS.
        config_path = Path(model_path) / "generation_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        eos = config.get("eos_token_id")
        if eos is None:
            model_config = json.loads((Path(model_path) / "config.json").read_text())
            eos = model_config.get("eos_token_id")
            if eos is None:
                eos = model_config.get("text_config", {}).get("eos_token_id")
        eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        tokenizer_eos = self.processor.tokenizer.eos_token_id
        if tokenizer_eos is not None:
            eos.append(tokenizer_eos)
        self.sampling = SamplingParams(
            temperature=0.0, max_tokens=MAX_NEW_TOKENS,
            stop_token_ids=sorted(set(eos)), skip_special_tokens=True,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )
        self.llm = LLM(
            model=model_path, dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=max_model_len, max_num_seqs=1,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": window_size},
            mm_processor_kwargs={"images_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}},
            gdn_prefill_backend="triton",
            generation_config="vllm", enforce_eager=enforce_eager,
        )

    def _infer_pages(self, pages: list[Path]) -> dict:
        if len(pages) == 1:
            output, elapsed = self._infer_window(pages, SINGLE_PAGE_PROMPT)
            return {
                "mode": "single_page",
                "n_pages": 1,
                "window_size": 1,
                "windows": [{"pages": [0], "elapsed_sec": elapsed}],
                "raw_output": output,
            }

        state = WindowState()
        outputs = []
        windows = []
        total_windows = (len(pages) + self.window_size - 1) // self.window_size
        for window_index, start in enumerate(range(0, len(pages), self.window_size)):
            window_pages = pages[start : start + self.window_size]
            page_ids = list(range(start, start + len(window_pages)))
            prompt = LONG_DOCUMENT_PROMPT.format(
                window_idx=window_index,
                window_number=window_index + 1,
                total_windows=total_windows,
                total_pages=len(pages),
                is_final_window=str(window_index == total_windows - 1).lower(),
                pages=", ".join(map(str, page_ids)),
                state=state.format(),
            )
            output, elapsed = self._infer_window(window_pages, prompt)
            state.update(output)
            outputs.append(output)
            windows.append({"pages": page_ids, "elapsed_sec": elapsed})
            print(
                f"Finished window {window_index + 1}/{total_windows} "
                f"(pages {start + 1}-{start + len(window_pages)}/{len(pages)}) "
                f"in {elapsed:.1f}s", flush=True,
            )

        return {
            "mode": "long_document",
            "n_pages": len(pages),
            "window_size": self.window_size,
            "windows": windows,
            "raw_output": "\n".join(filter(None, outputs)),
        }

    def _infer_window(self, pages: list[Path], prompt: str) -> tuple[str, float]:
        content = [{"type": "image", "image": str(page)} for page in pages]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": content},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        # Images stay open until synchronous generation has completed.
        with ExitStack() as stack:
            images = []
            for page in pages:
                source = stack.enter_context(Image.open(page))
                rgb = source.convert("RGB")
                stack.callback(rgb.close)
                images.append(rgb)
            start = time.perf_counter()
            result = self.llm.generate(
                [{"prompt": rendered, "multi_modal_data": {"image": images},
                  "mm_processor_kwargs": {"images_kwargs": {
                      "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                  }}}],
                self.sampling, use_tqdm=False,
            )[0]
            elapsed = time.perf_counter() - start
        output = result.outputs[0]
        if output.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {output.finish_reason}")
        stats = {
            "input_tokens": len(result.prompt_token_ids or []),
            "output_tokens": len(output.token_ids),
            "generate_wall_sec": elapsed,
            "output_tokens_per_generate_sec": len(output.token_ids) / elapsed if elapsed else None,
            "finish_reason": output.finish_reason,
            "stop_reason": output.stop_reason,
        }
        self.stats.append(stats)
        print("[VLLM] " + json.dumps(stats), flush=True)
        if output.finish_reason == "length":
            print("[WARNING] Output reached a length limit; parsing may be incomplete.", flush=True)
        return sanitize_output(output.text), elapsed

    def infer(self, pages: list[Path]) -> dict:
        self.stats = []
        result = self._infer_pages(pages)
        for window, stats in zip(result["windows"], self.stats):
            window.update(stats)
        result["backend"] = "vllm"
        result["vllm_version"] = "0.22.1"
        result["gdn_prefill_backend"] = "triton"
        result["complete_generation"] = all(s["finish_reason"] == "stop" for s in self.stats)
        return result


LABEL_START_RE = re.compile(
    r"^(<\|continued\|>)?(<\|title\|>|<\|text\|>|<\|table\|>|<\|image\|>|"
    r"<\|figure\|>|<\|chart\|>|<\|formula\|>|<\|caption\|>|"
    r"<\|code\|>|<\|pseudocode\|>|<\|chemistry\|>|<\|music\|>|<\|end\|>)"
)
LABEL_START_ANY_RE = re.compile(
    r"(?=(?:<\|continued\|>)?(?:<\|title\|>|<\|text\|>|<\|table\|>|<\|image\|>|"
    r"<\|figure\|>|<\|chart\|>|<\|formula\|>|<\|caption\|>|"
    r"<\|code\|>|<\|pseudocode\|>|<\|chemistry\|>|<\|music\|>|<\|end\|>))"
)
TAG_RE = re.compile(r"^<\|([^|]+)\|>")
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
BBOX_RE = re.compile(
    rf"<\|loc\|>\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*"
    rf"({NUMBER})\s*,\s*({NUMBER})"
)
PAGE_RE = re.compile(rf"<\|loc\|>\s*({NUMBER})")
METADATA_RE = re.compile(rf"<\|(?:loc|ref)\|>\s*{NUMBER}(?:\s*,\s*{NUMBER}){{0,4}}")
TITLE_RE = re.compile(
    r"^<\|title\|><\|level\|>(?P<level>-?\d+)<\|text\|>"
    r"(?P<text>.*?)(?=<\|loc\|>|<\|ref\|>|<\|truncated\|>|$)",
    re.DOTALL,
)
CONTROL_RE = re.compile(r"<\|(end|continued|truncated)\|>")


def split_logical_lines(text: str) -> list[str]:
    logical_lines = []
    current_parts = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        starts = [match.start() for match in LABEL_START_ANY_RE.finditer(stripped)]
        title_payload = re.match(r"^<\|title\|><\|level\|>-?\d+(<\|text\|>)", stripped)
        if title_payload:
            starts = [start for start in starts if start != title_payload.start(1)]
        starts = [
            start
            for start in sorted(set(starts))
            if stripped[max(0, start - len("<|continued|>")) : start] != "<|continued|>"
        ]
        pieces = [stripped]
        if starts and starts[0] == 0:
            boundaries = starts + [len(stripped)]
            pieces = [
                stripped[boundaries[index] : boundaries[index + 1]].strip()
                for index in range(len(boundaries) - 1)
                if stripped[boundaries[index] : boundaries[index + 1]].strip()
            ]
        if len(pieces) > 1:
            if current_parts:
                logical_lines.append("\n".join(current_parts))
                current_parts = []
            logical_lines.extend(pieces)
        elif LABEL_START_RE.match(pieces[0]):
            if current_parts:
                logical_lines.append("\n".join(current_parts))
            current_parts = [pieces[0]]
        elif current_parts:
            current_parts.append(raw_line)
    if current_parts:
        logical_lines.append("\n".join(current_parts))
    return logical_lines


def clean_content(text: str) -> str:
    text = CONTROL_RE.sub("", text)
    text = METADATA_RE.sub("", text)
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    return html.unescape(text).strip()


def parse_dsl(raw_output: str) -> list[dict]:
    events = []
    for raw_line in split_logical_lines(raw_output):
        line = raw_line.strip()
        continued = line.startswith("<|continued|>")
        if continued:
            line = line[len("<|continued|>") :]
        if line == "<|end|>":
            continue
        locs = [
            tuple(int(float(match.group(index))) for index in range(1, 6))
            for match in BBOX_RE.finditer(line)
        ]
        page_ids = [int(float(match.group(1))) for match in PAGE_RE.finditer(line)]
        title_match = TITLE_RE.match(line)
        if title_match:
            events.append(
                {
                    "type": "title",
                    "level": int(title_match.group("level")),
                    "text": clean_content(title_match.group("text")),
                    "locs": locs,
                    "page_ids": page_ids,
                    "continued": continued,
                    "truncated": "<|truncated|>" in line,
                }
            )
            continue
        tag_match = TAG_RE.match(line)
        if not tag_match:
            continue
        kind = tag_match.group(1)
        if kind == "title":
            level = 1
        elif kind not in {
            "text",
            "table",
            "image",
            "figure",
            "chart",
            "formula",
            "caption",
            "code",
            "pseudocode",
            "chemistry",
            "music",
        }:
            continue
        else:
            level = None
        events.append(
            {
                "type": "figure" if kind == "image" else kind,
                "level": level,
                "text": clean_content(line[len(f"<|{kind}|>") :]),
                "locs": locs,
                "page_ids": page_ids,
                "continued": continued,
                "truncated": "<|truncated|>" in line,
            }
        )
    return events


def merge_tables(parts: list[str]) -> str:
    parts = [
        segment.strip()
        for part in parts
        for segment in re.split(r"\s*<break>\s*", part or "")
        if segment.strip()
    ]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    rows = []
    for part in parts:
        table_rows = re.findall(r"<tr\b.*?</tr>", part, flags=re.IGNORECASE | re.DOTALL)
        rows.extend(table_rows or [f"<tr><td>{part}</td></tr>"])
    return "<table>" + "".join(rows) + "</table>"


def merge_continuations(events: list[dict]) -> list[dict]:
    merged = []
    previous = None
    mergeable = {"text", "table", "caption", "formula"}
    for source in events:
        event = dict(source)
        for key in ("locs", "page_ids"):
            if key in event:
                event[key] = list(event[key])
        kind = event["type"]
        if kind == "end":
            previous = None
            continue
        # Only the immediately preceding block can supply a continuation.
        if (
            previous is not None
            and kind in mergeable
            and previous["type"] == kind
            and previous.get("truncated")
            and event.get("continued")
        ):
            previous["parts"].append(event.get("text", ""))
            for key in ("locs", "page_ids"):
                if key in event:
                    previous.setdefault(key, []).extend(event[key])
            previous["text"] = (
                merge_tables(previous["parts"])
                if kind == "table" else "".join(previous["parts"])
            )
            previous["truncated"] = bool(event.get("truncated"))
            previous["continued_chain"] = True
        else:
            event["parts"] = [event.get("text", "")]
            merged.append(event)
            previous = event
    return merged


def strip_formula_delimiters(text: str) -> str:
    text = str(text or "").strip()
    pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    changed = True
    while text and changed:
        changed = False
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                changed = True
                break
    for left, _ in pairs:
        if text.startswith(left) and len(text) > len(left):
            return text[len(left):].strip()
    return text


def _trim_inline_math_spaces(markdown: str) -> str:
    # Consume complete formulas so adjacent delimiters cannot pair across prose.
    pattern = re.compile(
        r"(?<!\\)\$\$[\s\S]*?\$\$"
        r"|(?<![\\$])\$(?!\$)(?P<math>(?:\\[^\r\n]|[^$\\\r\n])+)\$(?!\$)"
    )
    def replace(match):
        math = match.group("math")
        padded = re.fullmatch(
            r"[^\S\r\n]+((?:\\[^\r\n]|[^$\\\r\n])*?)[^\S\r\n]+", math or ""
        )
        return "$" + padded[1] + "$" if padded and padded[1].strip() else match.group(0)
    return pattern.sub(replace, markdown)


def to_markdown(
    raw_output: str,
    visual_renderer: Optional[Callable[[dict, int], Optional[str]]] = None,
) -> str:
    blocks = []
    visual_index = 0
    for event in merge_continuations(parse_dsl(raw_output)):
        kind = event["type"]
        content = event["text"]
        block = None
        if kind == "title" and content:
            level = max(1, min(int(event.get("level") or 1), 6))
            block = f"{'#' * level} {content}"
        elif kind in {"text", "caption", "code", "pseudocode", "chemistry", "music"}:
            block = content or None
        elif kind == "table":
            block = merge_tables([content]) or None
        elif kind == "formula":
            formula = strip_formula_delimiters(content)
            block = f"$${formula}$$" if formula else None
        elif kind in {"figure", "chart"}:
            block = visual_renderer(event, visual_index) if visual_renderer else None
            visual_index += 1
        if block:
            blocks.append(block.strip())
    return _trim_inline_math_spaces(re.sub(r"\n{3,}", "\n\n", "\n\n".join(blocks)).strip())


def to_hierarchy_markdown(raw_output: str) -> str:
    titles = [
        event
        for event in parse_dsl(raw_output)
        if event["type"] == "title" and event["text"]
    ]
    lines = ["## 文档目录", ""]
    if not titles:
        return "\n".join(lines + ["_未识别到标题层级_"])
    level_stack = []
    for event in titles:
        level = max(1, min(int(event.get("level") or 1), 7))
        while level_stack and level_stack[-1] >= level:
            level_stack.pop()
        title = re.sub(
            r"([\\`*_[\]<>])", r"\\\1", event["text"].replace("\n", " ").strip()
        )
        page_ids = event.get("page_ids") or []
        page = f" ····· {page_ids[0] + 1}" if page_ids else ""
        label = f"**{title}**" if not level_stack else title
        lines.append(f"{'  ' * len(level_stack)}- {label}{page}")
        level_stack.append(level)
    return "\n".join(lines)


def postprocess(
    raw_output: str, pages: list[Path], output_dir: Path, mode: str
) -> tuple[str, str]:
    image_dir = output_dir / "images"

    def render_visual(event: dict, visual_index: int) -> Optional[str]:
        rendered = []
        for location_index, location in enumerate(event["locs"]):
            page_index, x1, y1, x2, y2 = location
            if not 0 <= page_index < len(pages):
                continue
            with Image.open(pages[page_index]) as source:
                width, height = source.size
                box = (
                    max(0, min(int(x1 / 1000 * width), width)),
                    max(0, min(int(y1 / 1000 * height), height)),
                    max(0, min(int(x2 / 1000 * width), width)),
                    max(0, min(int(y2 / 1000 * height), height)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                image_dir.mkdir(parents=True, exist_ok=True)
                filename = f"page_{page_index + 1:04d}_{event['type']}_{visual_index}_{location_index}.png"
                source.crop(box).save(image_dir / filename)
                rendered.append(f"![{event['type']}](images/{filename})")
        return "\n\n".join(rendered) or None

    markdown = to_markdown(raw_output, render_visual)
    hierarchy = to_hierarchy_markdown(raw_output) if mode == "long_document" else ""
    return markdown, hierarchy


def write_outputs(result: dict, output_dir: Path) -> None:
    result["postprocess_version"] = POSTPROCESS_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_output.dsl").write_text(result["raw_output"], encoding="utf-8")
    (output_dir / "markdown.md").write_text(result["markdown"], encoding="utf-8")
    (output_dir / "hierarchy.md").write_text(
        result["hierarchy_markdown"], encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Logics-Parsing-V3 local single-GPU vLLM inference.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()
    if args.window_size < 1:
        parser.error("--window_size must be >= 1")
    if args.max_model_len <= MAX_NEW_TOKENS:
        parser.error(f"--max_model_len must exceed {MAX_NEW_TOKENS} to leave input space")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu_memory_utilization must be between 0 and 1")
    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        parser.error("--model_path must be a downloaded HF model directory with config.json")
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="logics_vllm_") as tmp:
        pages = prepare_input(input_path, Path(tmp))
        effective_window = 1 if len(pages) == 1 else args.window_size
        engine = DocumentParser(
            str(model_path), window_size=effective_window,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        result = engine.infer(pages)
        result["markdown"], result["hierarchy_markdown"] = postprocess(
            result["raw_output"], pages, output_dir, result["mode"],
        )
        result["input_path"] = str(input_path)
        result["max_model_len"] = args.max_model_len
        result["max_new_tokens"] = MAX_NEW_TOKENS
        write_outputs(result, output_dir)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
