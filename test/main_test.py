from liberies_test import *
 
from app.main import app

client = TestClient(app)


def test_chat_endpoint():
    """Test the chat endpoint with a mocked model response."""
    mock_model = MagicMock()
    mock_model.chat.return_value = "Mocked response"

    with patch("app.main.model", mock_model):
        response = client.post(
            "/",
            json={"query": "Hello"}
        )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Mocked response"
    }

    mock_model.chat.assert_called_once_with("Hello")


def test_chat_error():
    """Test the chat endpoint when the model raises an exception."""
    mock_model = MagicMock()

    mock_model.chat.side_effect = Exception("Model failed")

    with patch("app.main.model", mock_model):
        response = client.post(
            "/",
            json={"query": "Hello"}
        )
    print(response.status_code)

    assert response.status_code == 500
    mock_model.chat.assert_called_once_with("Hello")