import re
from enum import StrEnum
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import logging

logger = logging.getLogger(__name__)


class Language(StrEnum):
    python = ".py"
    javascript = ".js"
    typescript = ".ts"
    jsx = ".jsx"
    tsx = ".tsx"
    go = ".go"
    bash = ".sh"
    other = ".txt"

    @classmethod
    def _missing_(cls, val):
        return cls.other


@dataclass
class CodeResponse:
    source: str
    lang: Language
    filename: Path

    def write(self, folder: Path = Path("./output")):
        (folder / self.filename).parent.mkdir(exist_ok=True, parents=True)
        with open(folder / self.filename, "w") as f:
            f.write(self.source)


def get_code_blocks(response) -> list[str]:
    return re.findall(r"```((?!.*[\'\"]\`\`\`).*?)```", response, re.DOTALL)


def extract_code_blocks(response: str) -> list[CodeResponse]:
    code_responses = []
    for content in get_code_blocks(response):
        codeblock = content[1].split("\n")
        language = get_language(codeblock)
        filename = get_filename(content, language)
        code_responses.append(
            CodeResponse("\n".join(codeblock[1:]), language, filename)
        )
    return code_responses


def get_language(block: list[str]) -> Language:
    try:
        language_line = block[0]
        return Language(language_line.strip())
    except (IndexError, ValueError, AttributeError) as exc:
        logger.error("could not get programming language", exc_info=exc)
        raise


def get_filename(block: str, language: Language) -> Path:
    error = None
    filename = ""

    try:
        filename = block[0].split("\n")[-1].strip()
    except (IndexError, ValueError, AttributeError) as exc:
        error = exc

    try:
        for line in block[1].split("\n")[1:]:
            if line.startswith("#"):
                filename = line
    except (IndexError, ValueError, AttributeError) as exc:
        error = exc

    if filename.startswith("#"):
        filename = filename[1:].strip()

    if filename.endswith(":"):
        filename = filename[:-1].strip()

    if not filename:
        logger.info("could not get filename, generating a random one", exc_info=error)
        filename = f"{uuid4()}{language.value}"

    return Path(filename)
