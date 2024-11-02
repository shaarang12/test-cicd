import sys
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


def test_summarize_endpoint_success(client, mock_gemini):
    mock_gemini.generate_content.return_value = create_mock_response("This is a shorter summary.")

    test_text = {
        "text": "This is a long text that needs to be summarized. " * 5
    }

    response = client.post('/summarize',
                           data=json.dumps(test_text),
                           content_type='application/json')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "summary" in data
    assert "original_length" in data
    assert "summary_length" in data


def test_summarize_endpoint_missing_text(client):
    response = client.post('/summarize',
                           data=json.dumps({}),
                           content_type='application/json')

    assert response.status_code == 400
    assert b"No text provided" in response.data


@patch('app.model.generate_content')
def test_api_error_handling(mock_generate, client):
    mock_generate.side_effect = Exception("API Error")

    response = client.post('/query',
                           data=json.dumps({"query": "test"}),
                           content_type='application/json')
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data

    response = client.post('/summarize',
                           data=json.dumps({"text": "test"}),
                           content_type='application/json')
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data


def test_malformed_json_handling(client):
    response = client.post('/query',
                           data="{invalid json}",
                           content_type='application/json')
    assert response.status_code == 500

    response = client.post('/summarize',
                           data="{invalid json}",
                           content_type='application/json')
    assert response.status_code == 500


@patch('google.generativeai.GenerativeModel')
def test_evaluate_query_endpoint(mock_model, client):
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = create_mock_response("Paris is the capital of France")
    mock_model.return_value = mock_instance

    response = client.get('/evaluate-query')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "matches" in data
    assert "total_cases" in data


@patch('google.generativeai.GenerativeModel')
def test_evaluate_summary_endpoint(mock_model, client):
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = create_mock_response("Short summary")
    mock_model.return_value = mock_instance

    response = client.get('/evaluate-summary')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "successful_summaries" in data
    assert "total_cases" in data