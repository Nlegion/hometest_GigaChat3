from unittest.mock import patch


def test_api_health_model_loaded(test_client, mock_model_handler):
    """Тест health check при загруженной модели."""
    with patch('app.main.model_handler', mock_model_handler):
        response = test_client.get('/api/health')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert 'model_info' in data
        assert data['model_info'] is not None


def test_api_health_model_not_loaded(test_client):
    """Тест health check при незагруженной модели."""
    with patch('app.main.model_handler', None):
        response = test_client.get('/api/health')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'degraded'
        assert data['model_loaded'] is False
        assert data['model_info'] is None


def test_api_health_model_info_content(test_client, mock_model_handler):
    """Тест содержимого model_info в health check."""
    with patch('app.main.model_handler', mock_model_handler):
        response = test_client.get('/api/health')
        data = response.json()

        model_info = data['model_info']
        assert 'loaded' in model_info
        assert 'llama_server_url' in model_info
        assert 'model_name' in model_info
        assert 'system_prompt_type' in model_info
        assert 'use_gpu' in model_info
