import click
from llm import LogMessage, Model, Prompt, Response, hookimpl
import google.generativeai as palm
from llm.errors import NeedsKeyException
import json


@hookimpl
def register_models(register):
    register(Palm("chat-bison-001"), aliases=("palm", "palm2"))


@hookimpl
def register_commands(cli):
    @cli.group(name="palm")
    def palm_():
        "Commands for working directly with PaLM"

    @palm_.command()
    @click.option("--key", help="PaLM API key")
    def models(key):
        "List models available in the PaLM API"
        from llm.cli import get_key

        api_key = get_key(key, "palm", "PALM_API_KEY")
        palm.configure(api_key=api_key)

        models = palm.list_models()
        from pprint import pprint

        pprint(list(models))


class PalmResponse(Response):
    def __init__(self, prompt, model, key):
        self.key = key
        super().__init__(prompt, model, stream=False)

    def iter_prompt(self):
        kwargs = {"messages": self.prompt.prompt}
        if self.prompt.system:
            kwargs["context"] = self.prompt.system

        response = palm.chat(**kwargs)
        last = response.last
        self._debug = {}
        self._done = True
        # last can be None
        yield last or ""

    def to_log(self) -> LogMessage:
        return LogMessage(
            model=self.prompt.model.model_id,
            prompt=self.prompt.prompt,
            system=self.prompt.system,
            options=None,
            prompt_json=json.dumps(self.prompt.prompt_json)
            if self.prompt.prompt_json
            else None,
            response=self.text(),
            response_json={},
            chat_id=None,  # TODO
            debug_json=self._debug,
        )


class Palm(Model):
    needs_key = "palm"

    def __init__(self, model_id, key=None):
        self.model_id = model_id
        self.key = key

    def execute(self, prompt: Prompt, stream: bool) -> PalmResponse:
        key = self.get_key()
        if key is None:
            raise NeedsKeyException(
                "{} needs an API key, label={}".format(str(self), self.needs_key)
            )
        palm.configure(api_key=key)
        return PalmResponse(prompt, self, key=self.key)

    def __str__(self):
        return "PaLM 2: {}".format(self.model_id)
