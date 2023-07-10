import click
import google.generativeai as palm
import llm
from llm.errors import NeedsKeyException
from pprint import pprint


@llm.hookimpl
def register_models(register):
    register(Palm("chat-bison-001"), aliases=("palm", "palm2"))


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="palm")
    def palm_():
        "Commands for working directly with PaLM"

    @palm_.command()
    @click.option("--key", help="PaLM API key")
    def models(key):
        "List models available in the PaLM API"
        api_key = llm.get_key(key, "palm", "PALM_API_KEY")
        palm.configure(api_key=api_key)
        models = palm.list_models()
        pprint(list(models))


class Palm(llm.Model):
    needs_key = "palm"
    key_env_var = "PALM_API_KEY"

    class Response(llm.Response):
        def __init__(self, prompt, model, stream, key):
            super().__init__(prompt, model, stream)
            self.key = key

        def iter_prompt(self, prompt):
            palm.configure(api_key=self.model.get_key())
            kwargs = {"messages": prompt.prompt}
            if prompt.system:
                kwargs["context"] = prompt.system

            response = palm.chat(**kwargs)
            last = response.last
            self._debug = {}
            self._done = True
            # last can be None
            yield last or ""

    def __init__(self, model_id):
        self.model_id = model_id

    def __str__(self):
        return "PaLM 2: {}".format(self.model_id)
