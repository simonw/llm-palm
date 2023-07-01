from llm import Model, Prompt, Response, hookimpl
from llm.errors import NeedsKeyException
import requests


@hookimpl
def register_models(register):
    register(Palm("text-bison-001"), aliases=("palm", "palm2"))


class PalmResponse(Response):
    def __init__(self, prompt, model, key):
        self.key = key
        super().__init__(prompt, model, stream=False)

    def iter_prompt(self):
        url = (
            f"https://generativelanguage.googleapis.com/v1beta2/models/{self.prompt.model.model_id}:generateText"
            f"?key={self.key}"
        )
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"prompt": {"text": self.prompt.prompt}},
        )
        data = response.json()
        candidate = data["candidates"][0]
        self._debug = {"safetyRatings": candidate["safetyRatings"]}
        self._done = True
        yield candidate["output"]


class Palm(Model):
    needs_key = "palm"

    def __init__(self, model_id, key=None):
        self.model_id = model_id
        self.key = key

    def execute(self, prompt: Prompt, stream: bool) -> PalmResponse:
        # ignore stream, since we cannot stream
        if self.key is None:
            raise NeedsKeyException(
                "{} needs an API key, label={}".format(str(self), self.needs_key)
            )
        return PalmResponse(prompt, self, key=self.key)

    def __str__(self):
        return "Vertex Chat: {}".format(self.model_id)
