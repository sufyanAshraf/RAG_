from liberies_test import*
 
from app.dataBase import dataBase

def test_init_success():
    """Test PC initializes with the expected settings."""
    mock_PC = MagicMock()

    with patch("app.dataBase.Pinecone", return_value=mock_PC) as mock_db:
        db = dataBase(key="fake_api_key")

    assert db.key == "fake_api_key" 
    assert db.pc is mock_PC
    mock_db.assert_called_once_with(api_key="fake_api_key")

def test_init_requires_api_key():
    """Test Pinecone rejects a missing API key."""
    with pytest.raises(ValueError, match="API key is required for Pinecone."):
        dataBase(key=None)


# def test_create_index_success():
#     """Test creation of index """
#     mock_pc = MagicMock() 



