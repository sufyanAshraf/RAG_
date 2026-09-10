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


def test_create_index_success():
    """Test creation of index when it does not already exist."""
    mock_pc = MagicMock()
    mock_pc.has_index.return_value = False

    with patch("app.dataBase.Pinecone", return_value=mock_pc):
        db = dataBase(key="fake_api_key")
        pc, flag = db.create_index(index_name="fake_name")

    assert pc is mock_pc
    assert flag is False
    mock_pc.create_index_for_model.assert_called_once_with(
        name="fake_name",
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"},
        },
    )

def test_Existance_index_success():
    """Test existance of index."""
    mock_pc = MagicMock()
    mock_pc.has_index.return_value = True

    with patch("app.dataBase.Pinecone", return_value=mock_pc):
        db = dataBase(key="fake_api_key")
        pc, flag = db.create_index(index_name="fake_name")

    assert pc is mock_pc
    assert flag is True
    mock_pc.has_index.assert_called_once_with("fake_name")