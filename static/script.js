let chatHistory = [];
let isGenerating = false;
let eventSource = null;

const chatArea = document.getElementById('chatArea');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const saveBtn = document.getElementById('saveBtn');
const loadBtn = document.getElementById('loadBtn');
const loadFile = document.getElementById('loadFile');
const loadingSpinner = document.getElementById('loadingSpinner');
const statusText = document.getElementById('statusText');
const status = document.getElementById('status');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const streamingToggle = document.getElementById('streaming');

temperatureSlider.addEventListener('input', (e) => {
    temperatureValue.textContent = e.target.value;
});

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

resetBtn.addEventListener('click', resetChat);
saveBtn.addEventListener('click', saveHistory);
loadBtn.addEventListener('click', () => loadFile.click());
loadFile.addEventListener('change', loadHistory);

checkHealth();

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.model_loaded) {
            status.textContent = 'Статус: Модель загружена и готова';
            status.className = 'small text-success';
        } else {
            status.textContent = 'Статус: Модель не загружена';
            status.className = 'small text-danger';
        }
    } catch (error) {
        status.textContent = 'Статус: Ошибка подключения';
        status.className = 'small text-danger';
        console.error('Health check failed:', error);
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isGenerating) {
        return;
    }

    const userMessage = {
        role: 'user',
        content: message
    };

    chatHistory.push(userMessage);
    addMessageToChat('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    isGenerating = true;
    sendBtn.disabled = true;
    loadingSpinner.classList.remove('d-none');
    statusText.textContent = 'Генерация ответа...';

    const systemPromptType = document.getElementById('systemPrompt').value;
    const temperature = parseFloat(temperatureSlider.value);
    const maxTokens = parseInt(document.getElementById('maxTokens').value);
    const stream = streamingToggle.checked;

    try {
        if (stream) {
            await sendStreamingRequest(userMessage, systemPromptType, temperature, maxTokens);
        } else {
            await sendRegularRequest(userMessage, systemPromptType, temperature, maxTokens);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToChat('error', `Ошибка: ${error.message}`);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        loadingSpinner.classList.add('d-none');
        statusText.textContent = '';
    }
}

async function sendRegularRequest(userMessage, systemPromptType, temperature, maxTokens) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            messages: chatHistory,
            temperature: temperature,
            max_tokens: maxTokens,
            stream: false,
            system_prompt_type: systemPromptType,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Ошибка при генерации ответа');
    }

    const data = await response.json();
    const assistantMessage = {
        role: 'assistant',
        content: data.content
    };

    chatHistory.push(assistantMessage);
    addMessageToChat('assistant', data.content);
}

async function sendStreamingRequest(userMessage, systemPromptType, temperature, maxTokens) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            messages: chatHistory,
            temperature: temperature,
            max_tokens: maxTokens,
            stream: true,
            system_prompt_type: systemPromptType,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Ошибка при генерации ответа');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessageId = null;
    let fullContent = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                    continue;
                }

                try {
                    const parsed = JSON.parse(data);
                    if (parsed.error) {
                        throw new Error(parsed.error);
                    }
                    if (parsed.content) {
                        fullContent += parsed.content;
                        if (assistantMessageId === null) {
                            assistantMessageId = addMessageToChat('assistant', '');
                        }
                        updateMessageContent(assistantMessageId, fullContent);
                    }
                } catch (e) {
                    if (data !== '[DONE]') {
                        console.error('Error parsing SSE data:', e, data);
                    }
                }
            }
        }
    }

    const assistantMessage = {
        role: 'assistant',
        content: fullContent
    };
    chatHistory.push(assistantMessage);
}

function addMessageToChat(role, content) {
    const welcomeMessage = chatArea.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message mb-3 ${role === 'user' ? 'user-message' : 'assistant-message'}`;
    
    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role small text-muted mb-1';
    roleLabel.textContent = role === 'user' ? 'Вы' : 'GigaChat3';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (role === 'error') {
        contentDiv.className += ' text-danger';
    }
    contentDiv.textContent = content;
    
    messageDiv.appendChild(roleLabel);
    messageDiv.appendChild(contentDiv);
    chatArea.appendChild(messageDiv);
    
    chatArea.scrollTop = chatArea.scrollHeight;
    
    return messageDiv;
}

function updateMessageContent(messageDiv, content) {
    const contentDiv = messageDiv.querySelector('.message-content');
    if (contentDiv) {
        contentDiv.textContent = content;
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

function resetChat() {
    if (isGenerating) {
        return;
    }

    if (confirm('Вы уверены, что хотите начать новый диалог?')) {
        chatHistory = [];
        chatArea.innerHTML = '<div class="welcome-message text-center text-muted"><p>Добро пожаловать! Начните диалог, отправив сообщение.</p></div>';
        
        fetch('/api/reset', { method: 'POST' })
            .catch(error => console.error('Error resetting chat:', error));
    }
}

function saveHistory() {
    const dataStr = JSON.stringify(chatHistory, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat_history_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function loadHistory(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const loadedHistory = JSON.parse(e.target.result);
            if (Array.isArray(loadedHistory)) {
                chatHistory = loadedHistory;
                chatArea.innerHTML = '';
                chatHistory.forEach(msg => {
                    addMessageToChat(msg.role, msg.content);
                });
            } else {
                alert('Неверный формат файла истории');
            }
        } catch (error) {
            alert('Ошибка при загрузке истории: ' + error.message);
        }
    };
    reader.readAsText(file);
    loadFile.value = '';
}
