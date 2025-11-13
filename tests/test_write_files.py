import pytest
from pathlib import Path

from llm_agent.parse.code import (
    CodeResponse,
    Language,
    extract_code_blocks,
    get_code_blocks,
    get_filename,
    get_language,
)


FIXTURE_FILE = Path(__file__).parent / "response.md"


def test_code_response_write(code_response):
    code_response.write()


def test_get_filename(content):
    code = get_code_blocks(content)[0]
    assert get_filename(code) == Path("llm_agent/output.py")


def test_get_code_blocks(content):
    codeblocks = get_code_blocks(content)
    assert len(codeblocks) == 1


def test_get_language(content):
    codeblocks = get_code_blocks(content)
    assert get_language(codeblocks[0][1].split("\n")) == Language.python


@pytest.fixture
def content():
    with open(FIXTURE_FILE) as f:
        return f.read()


@pytest.fixture
def code_response():
    return CodeResponse(
        source="""print("Hello World")""",
        lang=Language.python,
        filename=Path("tests/output/hello_world.py"),
    )
