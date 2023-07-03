import ast
from click.testing import CliRunner
from unittest.mock import patch, Mock
from llm.cli import cli
from llm_palm import PalmResponse, Prompt
import os


@patch("llm_palm.palm")
def test_palm_response(mock_palm):
    mock_response = Mock()
    mock_response.last = "hello"
    mock_palm.chat.return_value = mock_response

    r = PalmResponse(Prompt("hello", ""), "model", "key")
    items = list(r.iter_prompt())

    mock_palm.chat.assert_called_with(messages="hello")

    assert items == ["hello"]


@patch.dict(os.environ, {"PALM_API_KEY": "X"})
@patch("llm_palm.palm")
def test_palm_models(mock_palm):
    fake_models = [
        {"name": "'models/chat-bison-001"},
        {"name": "models/text-bison-001"},
    ]
    # Mock palm.list_models()
    mock_palm.list_models.return_value = fake_models
    runner = CliRunner()
    result = runner.invoke(cli, ["palm", "models"])
    assert result.exit_code == 0, result.output
    assert ast.literal_eval(result.output) == fake_models


@patch.dict(os.environ, {"PALM_API_KEY": "X"})
@patch("llm_palm.palm")
def test_palm_prompt(mock_palm):
    mock_response = Mock()
    mock_response.last = "🐶🐶"
    mock_palm.chat.return_value = mock_response
    runner = CliRunner()
    result = runner.invoke(cli, ["two dog emoji", "-m", "palm"])
    assert result.exit_code == 0, result.output
    assert result.output == "🐶🐶\n"
