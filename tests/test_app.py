import os
import pytest
from app import app
import json
from unittest.mock import patch, Mock, MagicMock

os.environ['GEMINI_API_KEY'] = 'test_key'


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_mock_response(text):
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


@pytest.fixture
def mock_gemini():
    with patch('google.generativeai.GenerativeModel') as mock:
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = create_mock_response("Generic response")
        mock.return_value = mock_instance
        yield mock_instance


def test_query_endpoint_success(client, mock_gemini):
    mock_gemini.generate_content.return_value = create_mock_response("The capital of France is Paris.")

    response = client.post('/query',
                           data=json.dumps({"query": "What is the capital of France?"}),
                           content_type='application/json')

    assert response.status_code == 200
    assert b"Paris" in response.data


def test_query_endpoint_missing_query(client):
    response = client.post('/query',
                           data=json.dumps({}),
                           content_type='application/json')

    assert response.status_code == 400
    assert b"No query provided" in response.data


@patch('google.generativeai.GenerativeModel')
def test_evaluate_query_endpoint(mock_model, client):
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = create_mock_response("Paris is the capital of France")
    mock_model.return_value = mock_instance

    response = client.get('/evaluate')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "matches" in data
    assert "total_cases" in data

