from liberies_test import *

from app.models import GroqModel


def test_init_success():
    """Test GroqModel initializes with the expected settings."""
    mock_llm = MagicMock()

    with patch("app.models.ChatGroq", return_value=mock_llm) as mock_chat_groq:
        model = GroqModel(key="fake_api_key", model_name="llama-3.1-8b-instant")

    assert model.key == "fake_api_key"
    assert model.model_name == "llama-3.1-8b-instant"
    assert model.llm is mock_llm
    mock_chat_groq.assert_called_once_with(
        groq_api_key="fake_api_key",
        model_name="llama-3.1-8b-instant",
    )


def test_init_requires_api_key():
    """Test GroqModel rejects a missing API key."""
    with pytest.raises(ValueError, match="API key is required for GroqModel."):
        GroqModel(key=None)


def test_get_model_name():
    """Test the model name is returned as configured."""
    mock_llm = MagicMock()

    with patch("app.models.ChatGroq", return_value=mock_llm):
        model = GroqModel(key="fake_api_key")

    assert model.get_model_name() == "openai/gpt-oss-safeguard-20b"


def test_invoke_model_success():
    """Test the model invokes the LLM with the expected messages."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked model response"
    mock_llm.invoke.return_value = mock_response

    with patch("app.models.ChatGroq", return_value=mock_llm):
        model = GroqModel(key="fake_api_key")
        response = model.invoke_model(
            "Hello there",
            system_prompt="You are a helpful assistant.",
        )

    assert response == "Mocked model response"
    mock_llm.invoke.assert_called_once_with(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there"},
        ]
    )


def test_invoke_model_requires_prompt():
    """Test invoking the model without a prompt raises a validation error."""
    mock_llm = MagicMock()

    with patch("app.models.ChatGroq", return_value=mock_llm):
        model = GroqModel(key="fake_api_key")

    with pytest.raises(ValueError, match="Full prompt is required to invoke the model."):
        model.invoke_model("")


def test_invoke_model_error():
    """Test model invocation errors are surfaced to the caller."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Model interaction failed")

    with patch("app.models.ChatGroq", return_value=mock_llm):
        model = GroqModel(key="fake_api_key")

    with pytest.raises(Exception, match="Failed to invoke the model: Model interaction failed"):
        model.invoke_model("Hello")