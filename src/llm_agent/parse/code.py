import json
import re
from dataclasses import dataclass
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeResponse:
    source: str
    filename: Path

    def write(self, folder: Path = Path("./output")):
        (folder / self.filename).parent.mkdir(exist_ok=True, parents=True)
        with open(folder / self.filename, "w") as f:
            f.write(self.source)


def get_code_blocks(response) -> list[str]:
    return re.findall(r"```((?!.*[\'\"]\`\`\`).*?)```", response, re.DOTALL)


def extract_code_blocks(response: str) -> list[CodeResponse]:
    contents = {}
    for code_block in get_code_blocks(response):
        match = re.search(r"\{.*\}", code_block, re.DOTALL)
        if not match:
            continue
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            contents.update(data)
        except json.JSONDecodeError:
            print("Found a match, but it wasn't valid JSON.")

    code_responses = []
    for filename, file_content in contents.items():
        code_responses.append(CodeResponse(file_content, filename))
    return code_responses
