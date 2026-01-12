def test_api_reset(test_client):
    """Тест эндпоинта сброса контекста."""
    response = test_client.post('/api/reset')

    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert data['status'] == 'ok'
    assert 'message' in data
