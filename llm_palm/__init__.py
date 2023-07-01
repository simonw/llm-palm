from llm import Model, Prompt, Response, hookimpl
import google.generativeai as palm
from llm.errors import NeedsKeyException


@hookimpl
def register_models(register):
    register(Palm("chat-bison-001"), aliases=("palm", "palm2"))


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
        yield last


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
        return "Vertex Chat: {}".format(self.model_id)
