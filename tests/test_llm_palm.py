from unittest.mock import patch, Mock
from llm_palm import PalmResponse, Prompt


@patch("llm_palm.palm")
def test_palm_response(mock_palm):
    mock_response = Mock()
    mock_response.last = "hello"
    mock_palm.chat.return_value = mock_response

    r = PalmResponse(Prompt("hello", ""), "model", "key")
    items = list(r.iter_prompt())

    mock_palm.chat.assert_called_with(messages="hello")

    assert items == ["hello"]
