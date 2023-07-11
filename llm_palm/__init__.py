import click
import google.generativeai as palm
import llm
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

    def __init__(self, model_id):
        self.model_id = model_id

    def build_prompt_messages(self, prompt, conversation):
        if not conversation:
            return prompt
        messages = []
        for response in conversation.responses:
            messages.append(response.prompt.prompt)
            messages.append(response.text())
        messages.append(prompt)
        return messages

    def execute(self, prompt, stream, response, conversation):
        palm.configure(api_key=self.get_key())

        kwargs = {"messages": self.build_prompt_messages(prompt.prompt, conversation)}
        if prompt.system:
            kwargs["context"] = prompt.system

        palm_response = palm.chat(**kwargs)
        last = palm_response.last
        yield last or ""
        response._prompt_json = kwargs

    def __str__(self):
        return "PaLM 2: {}".format(self.model_id)
