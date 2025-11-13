```python
# llm_agent/output.py
import json
from pathlib import Path

from llm_agent.parse.code import extract_code_blocks, CodeResponse


class OutputFormat:
    def format(self, content: str) -> str:
        raise NotImplementedError

    def file_ext(self) -> str:
        raise NotImplementedError

    def enrich_prompt(self, system_prompt) -> str:
        return system_prompt

    def write(self, folder: Path, content: str):
        raise NotImplementedError


class MarkdownFormat(OutputFormat):
    def format(self, content: str) -> str:
        # Ensure at most 8 bullet points
        lines = content.strip().splitlines()
        bullets = [line for line in lines if line.startswith("-")]
        if len(bullets) > 8:
            bullets = bullets[:8] + ["- ..."]
        return "\n".join(bullets) if bullets else content

    def file_ext(self) -> str:
        return ".md"

    def enrich_prompt(self, system_prompt) -> str:
        return system_prompt + ("Output in structured Markdown; max 8 bullet points.")

    def write(self, folder: Path, content: str):
        filename = folder / f"output{self.file_ext()}"
        filename.write_text(content)


class PlainTextFormat(OutputFormat):
    def format(self, content: str) -> str:
        return content.strip()

    def file_ext(self) -> str:
        return ".txt"

    def write(self, folder: Path, content: str):
        filename = folder / f"output{self.file_ext()}"
        filename.write_text(content)


class JsonFormat(OutputFormat):
    def format(self, content: str) -> str:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"output": content}
        return json.dumps(data, indent=2)

    def file_ext(self) -> str:
        return ".json"

    def write(self, folder: Path, content: str):
        filename = folder / f"output{self.file_ext()}"
        filename.write_text(content)


class CodeFormat(OutputFormat):
    def format(self, content: str) -> str:
        return content

    def write(self, folder: Path, content: str):
        code_blocks = extract_code_blocks(content)
        for code_block in code_blocks:
            code_block.write(folder)

    def enrich_prompt(self, system_prompt) -> str:
        return system_prompt + CODE_FORMAT


CODE_FORMAT = """
CONSTRAINTS: 
    - Use code adhering to SOLID principles
    - Consider LTS versions of programming languages
OUTPUT: 
    - code block with console format describning the file tree of the solution
    - code block with the solution
    - the first line of the code block should contain a comment with the folder/fine_name, matching the file tree
    - the content of each file in the solution must be in a different code block
    - a code block for a setup.sh file with a command to install the dependencies of the code 
"""
```

In this updated code, the `write` method has been added to each output format class. This method writes the formatted content to a file in the specified folder. The `CodeFormat` class uses the `extract_code_blocks` function to parse the content into code blocks, and then writes each code block to a separate file in the specified folder.