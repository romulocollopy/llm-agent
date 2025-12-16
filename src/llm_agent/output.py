import json
from pathlib import Path

from llm_agent.parse.code import extract_code_blocks


class OutputFormat:
    def format(self, content: str) -> str:
        return content.strip()

    def enrich_prompt(self, system_prompt: tuple[str, ...]) -> tuple[str, ...]:
        return system_prompt

    def write(self, folder: Path, content: str, raw_response: str = ""):
        folder.mkdir(exist_ok=True, parents=True)
        filename = folder / f"output{self.file_ext()}"
        filename.write_text(content)

    def file_ext(self) -> str:
        raise NotImplementedError


class MarkdownFormat(OutputFormat):
    def file_ext(self) -> str:
        return ".md"

    def enrich_prompt(self, system_prompt: tuple[str, ...]) -> tuple[str, ...]:
        return system_prompt + (
            "Output in structured Markdown",
            # "create a maximun of 8 bullet points with the most relevant information.",
        )


class EmailFormat(OutputFormat):
    def file_ext(self) -> str:
        return ".rtf"

    def enrich_prompt(self, system_prompt: tuple[str, ...]) -> tuple[str, ...]:
        return system_prompt + ("Output in Rich Text Format (rtf)",)


class PlainTextFormat(OutputFormat):
    def file_ext(self) -> str:
        return ".txt"


class JsonFormat(OutputFormat):
    def format(self, content: str) -> str:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"output": content}
        return json.dumps(data, indent=2)


class CodeFormat(OutputFormat):
    def write(self, folder: Path, content: str, raw_response: str = ""):
        (folder / "raw_response.md").write_text(content)
        code_blocks = extract_code_blocks(content)
        for code_block in code_blocks:
            code_block.write(folder)

    def enrich_prompt(self, system_prompt: tuple[str, ...]) -> tuple[str, ...]:
        return system_prompt + CODE_FORMAT


CODE_FORMAT = (
    """CONSTRAINTS: 
    - Use code adhering to SOLID principles
    - Consider LTS versions of programming languages
    - include a setup.sh file with a command to install the dependencies of the code.""",
    """DESIRED OUTPUT: 
    - a json structure where the key is the file path relative to the root and the value is the content of the files.
""",
)
