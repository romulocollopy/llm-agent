import datetime
from itertools import chain
import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain.schema import AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from sqlalchemy import Engine, create_engine

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_agent.llm_models import ModelName
from llm_agent.roles import Role

from . import output as out

TASK_FILENAME = "task.md"
MODEL_DEFAULT = ModelName.GEMINI

SEPARATOR = "---BEGIN_AI_RESPONSE---"
USER_PROMPT_SEPARATOR = "---END_AI_RESPONSE---"

DEFAUL_SYSTEM_PROMPT = (
    "You are a {persona.value}.",
    "Be concise, avoid historical/contextual filler, and focus on the task.",
)


@dataclass
class LLMRunner:
    model: ModelName
    persona: Role
    output_format: out.OutputFormat
    llm: BaseChatModel = field(init=False)
    context_path: Path = field(default_factory=lambda: Path("./src"))
    engine: Engine = create_engine(url="sqlite:///sqlite.db")
    session_id: str = ""
    task_file: Path = field(default_factory=lambda: Path(TASK_FILENAME))

    def __post_init__(self):
        # self.llm = ChatOpenAI(model=self.model.value, temperature=0.2)
        self.llm = ChatGoogleGenerativeAI(model=self.model.value, temperature=0.2)

    def run(self):
        task = self.read_task()
        last_human_prompt = self.extract_last_human_prompt(task)
        prompt = self.prepare_pipeline()
        response = self.call_llm(prompt, last_human_prompt)
        self.write_output(response, task, last_human_prompt)

    def call_llm(self, prompt, task) -> AIMessage:
        chain = prompt | self.llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.get_chat_history(session_id),
            input_messages_key="question",
            history_messages_key="history",
        )
        config = {"configurable": {"session_id": self.session_id}}
        return chain_with_history.invoke(
            {"question": f"Task:\n{task}"}, config=RunnableConfig(config)
        )

    def prepare_pipeline(self) -> ChatPromptTemplate:
        system_prompt = self.output_format.enrich_prompt(self.system_prompt)
        context = self.load_context()

        return ChatPromptTemplate.from_messages(
            [
                *[("system", prompt) for prompt in system_prompt],
                MessagesPlaceholder(variable_name="history"),
                ("system", f"CONTEXT: {context}"),
                ("human", "{question}"),
            ]
        )

    def write_output(self, message: AIMessage, task: str, last_human_prompt: str):
        folder = self.output_folder(task)
        print(message)

        formatted = self.output_format.format(str(message.content))
        self.output_format.write(folder, formatted, message.content)

        # write params
        params = {
            "model": self.model.value,
            "persona": self.persona.value,
            "output_format": type(self.output_format).__name__,
            "context_path": str(self.context_path),
            "task": task,
        }
        paramfile = folder / "input_params.txt"
        paramfile.write_text(json.dumps(params, indent=2))

        # Append AI response to task file
        with open(self.task_file, "a") as f:
            f.write(f"\n{SEPARATOR}\n")
            f.write(message.content)
            f.write(f"\n{USER_PROMPT_SEPARATOR}\n")

    @property
    def system_prompt(self) -> tuple[str, ...]:
        return tuple([p.format(persona=self.persona) for p in DEFAUL_SYSTEM_PROMPT])

    def extract_last_human_prompt(self, task_content: str) -> str:
        parts = task_content.split(USER_PROMPT_SEPARATOR)
        return parts[-1].split(SEPARATOR)[0].strip() if parts else task_content.strip()

    def read_task(self) -> str:
        return Path(TASK_FILENAME).read_text()

    def load_context(self) -> str:
        gitignore = self.context_path / ".gitignore"
        ignore = {".git", "node_modules", ".env", "chromium"}

        if gitignore.exists() and gitignore.is_file():
            for l in gitignore.read_text().split("\n"):
                ignore.add(l)

        print(f"ignoring: {ignore}")
        if not self.context_path.exists():
            return ""

        context_data = ""
        for path in self.context_path.rglob("*"):
            if ignore & set(chain(*[p.parts for p in path.parents])):
                print(f"ignoring from set: {path}")
                continue
            if path.is_file():
                if path.name.startswith("."):
                    print(f"ignoring hidden: {path}")
                    continue
                try:
                    context_data += f"\n=== {path.relative_to(self.context_path)} ===\n"
                    context_data += (
                        path.read_text().replace("{", "{{").replace("}", "}}")
                    )
                except UnicodeDecodeError:
                    pass
        return context_data

    @staticmethod
    def summarize_task(task: str) -> str:
        # naive summary: first 10 words
        return " ".join(task.strip().split()[:10])

    def output_folder(self, task: str):
        task_summary = self.summarize_task(task)
        today = datetime.date.today().isoformat()
        folder = Path("output") / today / task_summary.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_chat_history(self, session_id: str):
        chat_message_history = SQLChatMessageHistory(
            session_id=session_id, connection=self.engine
        )
        # chat_message_history.add_user_message("Hello")
        # chat_message_history.add_ai_message("Hi")
        return chat_message_history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LangChain task with configurable model/persona/output"
    )
    parser.add_argument("--model", type=ModelName, default=MODEL_DEFAULT)
    parser.add_argument("--session_id", type=str)
    parser.add_argument("--persona", type=Role, required=True)
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "text", "json", "code"],
    )
    parser.add_argument(
        "--context", type=Path, default=None, help="Optional path for context files"
    )

    args = parser.parse_args()

    fmt_class = {
        "markdown": out.MarkdownFormat,
        "text": out.PlainTextFormat,
        "json": out.JsonFormat,
        "code": out.CodeFormat,
    }[args.format]()

    runner = LLMRunner(
        model=args.model,
        persona=args.persona,
        output_format=fmt_class,
        context_path=args.context,
        session_id=args.session_id,
    )
    runner.run()


if __name__ == "__main__":
    main()
