from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from llm_agent.main import MODEL_DEFAULT, LLMRunner
from llm_agent.output import CodeFormat
from llm_agent.parse.code import CodeResponse, get_code_blocks
from llm_agent.roles import Role

FIXTURE_FILE = Path(__file__).parent / "response.md"


def test_code_response_write(code_response):
    code_response.write()


def test_get_code_blocks(content):
    codeblocks = get_code_blocks(content)
    assert len(codeblocks) == 1


@pytest.fixture
def content():
    with open(FIXTURE_FILE) as f:
        return f.read()


@pytest.fixture
def code_response():
    return CodeResponse(
        source="""print("Hello World")""", filename=Path("tests/output/hello_world.py")
    )


def test_write_output(llm_runner: LLMRunner, ai_message: AIMessage):
    llm_runner.write_output(ai_message, "some task")


@pytest.fixture
def llm_runner():
    return LLMRunner(
        MODEL_DEFAULT,
        Role.DEVELOPER,
        context_path=Path("./my_context_folder"),
        output_format=CodeFormat(),
        api_key="mock",
    )


@pytest.fixture
def ai_message():
    """Returns a standard AIMessage instance for testing."""
    return AIMessage(
        content='I\'m sorry for the confusion, but as a text-based AI, I\'m unable to create a file structure directly. However, I can provide you with the content of each file in the desired JSON format. Here it is:\n\n```json\n{\n    "my_package/setup.py": "from setuptools import setup, find_packages\\n\\nsetup(\\n    name=\'my_package\',\\n    version=\'0.1\',\\n    packages=find_packages(),\\n    install_requires=[\\n        \'pytest\',\\n    ],\\n    extras_require={\\n        \'dev\': [\\n            \'ruff\',\\n        ],\\n    },\\n)",\n    "my_package/setup.sh": "#!/bin/bash\\npip install -e .[dev]",\n    "my_package/my_package/__init__.py": "from .my_module import add",\n    "my_package/my_package/my_module.py": "def add(obj1, obj2):\\n    return obj1.add(obj2)",\n    "my_package/tests/__init__.py": "",\n    "my_package/tests/test_my_module.py": "from my_package import add\\n\\nclass MyObject:\\n    def __init__(self, value):\\n        self.value = value\\n\\n    def add(self, other):\\n        return self.value + other.value\\n\\ndef test_add():\\n    obj1 = MyObject(1)\\n    obj2 = MyObject(2)\\n    assert add(obj1, obj2) == 3"\n}\n```\n\nThis JSON structure represents the content of each file in your package. You can create these files manually and copy the corresponding content from the JSON to each file.'
    )
