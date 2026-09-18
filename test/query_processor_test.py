from unittest.mock import MagicMock

from app.queryProcessor import QueryProcessor


def test_greeting_returns_royal_hotel_welcome_without_calling_model():
    model = MagicMock()

    result = QueryProcessor().process(model, "  Good morning! ")

    assert result["allowed"] is False
    assert result["category"] == "greeting"
    assert result["message"] == "Welcome to Royal Hotel. How can I help you?"
    model.invoke_model.assert_not_called()