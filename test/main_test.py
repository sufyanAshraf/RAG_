from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_endpoint_success():
    """Test the chat endpoint returns the model response for a valid request."""
    mock_db = MagicMock()
    mock_db.create_index.return_value = (MagicMock(), False)
    mock_pc = MagicMock()
    mock_db.create_index.return_value = (mock_pc, False)

    mock_read_data = MagicMock()
    mock_read_data.readjson.return_value = [{"text": "sample hotel data"}]

    mock_model = MagicMock()
    mock_model.invoke_model.return_value = "Mocked response"

    with (
        patch("app.main.read_api_key_from_config", return_value=("fake_groq_key", "fake_pinecone_key")),
        patch("app.main.dataBase", return_value=mock_db),
        patch("app.main.readData", return_value=mock_read_data),
        patch("app.main.create_embeddings", return_value=mock_pc),
        patch("app.main.pc_search", return_value=[{"content": "hotel info"}]),
        patch("app.main.getPrompt", return_value="compiled prompt"),
        patch("app.main.GroqModel", return_value=mock_model) as mock_groq_model,
    ):
        response = client.post("/", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.json() == {"response": "Mocked response"}
    mock_groq_model.assert_called_once_with("fake_groq_key")
    mock_model.invoke_model.assert_called_once_with(full_prompt="compiled prompt")


def test_chat_endpoint_error():
    """Test the chat endpoint surfaces model invocation exceptions."""
    mock_db = MagicMock()
    mock_db.create_index.return_value = (MagicMock(), True)

    mock_model = MagicMock()
    mock_model.invoke_model.side_effect = Exception("Model failed")

    with (
        patch("app.main.read_api_key_from_config", return_value=("fake_groq_key", "fake_pinecone_key")),
        patch("app.main.dataBase", return_value=mock_db),
        patch("app.main.pc_search", return_value=[{"content": "hotel info"}]),
        patch("app.main.getPrompt", return_value="compiled prompt"),
        patch("app.main.GroqModel", return_value=mock_model),
    ):
        with pytest.raises(Exception, match="Model failed"):
            client.post("/", json={"query": "Hello"})

    mock_model.invoke_model.assert_called_once_with(full_prompt="compiled prompt")