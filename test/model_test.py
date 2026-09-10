import pytest

from liberies_test import *
from app.ingession import Model 


def test_init():
    """Test the initialization of the Model class."""
    mock_client = MagicMock()

    with patch("app.model.genai.Client", return_value=mock_client) as mock_genai_client:
        model = Model(api_key="fake_api_key")
  
    assert model.api_key == "fake_api_key"
    assert model.model_name == "gemini-3.8-flash"

    mock_genai_client.assert_called_once_with(api_key="fake_api_key")

 
     
def test_init_error(): 
    """Test the initialization of the Model class.""" 

    with patch("app.model.genai.Client", side_effect=Exception("Client initialization failed")
    ):
        with pytest.raises(Exception, match="Client initialization failed" ):
            Model(api_key="fake_api_key")

def test_chat(): 
    """Test the chat method of the Model class."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked model response"
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.model.genai.Client", return_value=mock_client):
        model = Model(api_key="fake_api_key")
        response = model.chat("Hello")

    assert response == "Mocked model response"
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-3.8-flash",
        contents="Hello",
    )

def test_chat_error(): 
    """Test the chat method of the Model class when an error occurs."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Model interaction failed")

    with patch("app.model.genai.Client", return_value=mock_client):
        model = Model(api_key="fake_api_key")
        
        with pytest.raises(Exception, match="Model interaction failed" ):
            model.chat("Hello")