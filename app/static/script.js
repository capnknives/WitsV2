document.addEventListener('DOMContentLoaded', () => {
    const goalInput = document.getElementById('goal-input');
    const sendGoalBtn = document.getElementById('send-goal-btn');
    const conversationDiv = document.getElementById('conversation');
    const thinkingProcessDiv = document.getElementById('thinking-process');

    const sessionIdInput = document.getElementById('session-id-input');
    const newSessionBtn = document.getElementById('new-session-btn');
    const agentSelect = document.getElementById('agent-select');

    const modelSelect = document.getElementById('model-select');
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureValueSpan = document.getElementById('temperature-value');
    const updateParamsBtn = document.getElementById('update-params-btn');

    const memorySearchInput = document.getElementById('memory-search-input');
    const searchMemoryBtn = document.getElementById('search-memory-btn');
    const clearMemoryBtn = document.getElementById('clear-memory-btn');
    const memorySearchResultsDiv = document.getElementById('memory-search-results');
    const memorySessionIdDisplay = document.getElementById('memory-session-id-display');

    let currentSessionId = localStorage.getItem('witsNexusSessionId') || `web_session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
    localStorage.setItem('witsNexusSessionId', currentSessionId);
    sessionIdInput.value = currentSessionId;
    memorySessionIdDisplay.textContent = currentSessionId.substring(0,15) + "...";


    newSessionBtn.addEventListener('click', () => {
        currentSessionId = `web_session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
        localStorage.setItem('witsNexusSessionId', currentSessionId);
        sessionIdInput.value = currentSessionId;
        memorySessionIdDisplay.textContent = currentSessionId.substring(0,15) + "...";
        conversationDiv.innerHTML = ""; // Clear conversation
        thinkingProcessDiv.innerHTML = "<h3>Agent Thinking Process:</h3>"; // Clear thinking
        memorySearchResultsDiv.innerHTML = ""; // Clear memory results
        addMessageToLog({type: 'info', content: `New session started: ${currentSessionId}`}, thinkingProcessDiv);
        fetchAgentProfiles(); // Re-fetch profiles to reset agent selection potentially
        fetchSessionLLMParameters(); // Fetch params for the new session (will likely be default)
    });

    // --- Agent Selection ---
    async function fetchAgentProfiles() {
        try {
            const response = await fetch('/api/agents');
            if (!response.ok) throw new Error(`Failed to fetch agent profiles: ${response.statusText}`);
            const profiles = await response.json();
            
            agentSelect.innerHTML = ''; // Clear existing
            let previouslySelectedAgent = localStorage.getItem(`selectedAgent_${currentSessionId}`);

            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.name;
                option.textContent = profile.display_name || profile.name;
                option.title = profile.description || '';
                agentSelect.appendChild(option);
            });

            if (previouslySelectedAgent && profiles.some(p => p.name === previouslySelectedAgent)) {
                agentSelect.value = previouslySelectedAgent;
            } else if (profiles.length > 0) {
                // Try to get default from config if possible, or just select first
                // For now, just select the first one if no previous selection
                 previouslySelectedAgent = profiles[0].name; // Fallback to first
                 agentSelect.value = previouslySelectedAgent;
            }
            
            if (agentSelect.value) { // If an agent is selected (either previous or first)
                 await selectAgentForSession(agentSelect.value); // Inform backend and fetch its params
            }


        } catch (error) {
            console.error('Error fetching agent profiles:', error);
            addMessageToLog({ type: 'error', content: 'Failed to load agent profiles.' }, thinkingProcessDiv);
        }
    }

    agentSelect.addEventListener('change', async () => {
        const selectedProfileName = agentSelect.value;
        if (selectedProfileName) {
            await selectAgentForSession(selectedProfileName);
        }
    });

    async function selectAgentForSession(profileName) {
        try {
            const response = await fetch('/api/session/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, agent_profile_name: profileName })
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(`Failed to select agent: ${errData.detail || response.statusText}`);
            }
            const result = await response.json();
            localStorage.setItem(`selectedAgent_${currentSessionId}`, profileName); // Store selection per session
            addMessageToLog({ type: 'info', content: `Switched to agent: ${profileName}. ${result.message || ''}` }, thinkingProcessDiv);
            fetchSessionLLMParameters(); // Fetch parameters for the newly selected agent
        } catch (error) {
            console.error('Error selecting agent:', error);
            addMessageToLog({ type: 'error', content: `Error switching agent: ${error.message}` }, thinkingProcessDiv);
        }
    }

    // --- LLM Parameter Controls (for the current session's agent) ---
    temperatureSlider.addEventListener('input', () => {
        temperatureValueSpan.textContent = temperatureSlider.value;
    });

    async function fetchSessionLLMParameters() {
        try {
            const response = await fetch(`/api/session/parameters?session_id=${currentSessionId}`);
            if (!response.ok) {
                console.error('Failed to fetch session LLM parameters:', response.statusText);
                addMessageToLog({ type: 'error', content: 'Failed to load session LLM parameters.' }, thinkingProcessDiv);
                return;
            }
            const params = await response.json();
            temperatureSlider.value = params.temperature || 0.7;
            temperatureValueSpan.textContent = temperatureSlider.value;

            modelSelect.innerHTML = ''; // Clear existing options
            params.available_models.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName;
                if (modelName === params.model) {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
             if (params.active_agent_profile && agentSelect.value !== params.active_agent_profile) {
                // This can happen if default agent is used before selection
                // agentSelect.value = params.active_agent_profile; // Sync agent dropdown if needed
            }

        } catch (error) {
            console.error('Error fetching session LLM parameters:', error);
            addMessageToLog({ type: 'error', content: 'Error loading session LLM parameters.' }, thinkingProcessDiv);
        }
    }

    updateParamsBtn.addEventListener('click', async () => {
        const payload = {
            session_id: currentSessionId,
            model: modelSelect.value,
            temperature: parseFloat(temperatureSlider.value)
        };
        addMessageToLog({type:'info', content: `Updating LLM params for agent ${agentSelect.options[agentSelect.selectedIndex].text}...`}, thinkingProcessDiv);
        try {
            const response = await fetch('/api/config/parameters', { // This endpoint now expects session_id
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok) {
                addMessageToLog({ type: 'error', content: `Failed to update params: ${result.detail || 'Unknown error'}` }, thinkingProcessDiv);
            } else {
                addMessageToLog({ type: 'info', content: result.message || 'Parameters updated.' }, thinkingProcessDiv);
            }
        } catch (error) {
            console.error('Error updating LLM parameters:', error);
            addMessageToLog({ type: 'error', content: 'Error updating LLM parameters.' }, thinkingProcessDiv);
        }
    });

    // --- Chat Functionality ---
    sendGoalBtn.addEventListener('click', sendGoal);
    goalInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendGoal();
    });

    async function sendGoal() {
        const goal = goalInput.value.trim();
        if (!goal) return;

        addMessageToLog({ type: 'user_goal', content: goal }, conversationDiv);
        goalInput.value = '';
        thinkingProcessDiv.innerHTML = '<h3>Agent Thinking Process:</h3>';

        const currentAgentProfile = agentSelect.value; // Get currently selected agent profile

        try {
            const response = await fetch('/api/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    goal: goal,
                    session_id: currentSessionId,
                    agent_profile_name: currentAgentProfile // Send the selected agent profile
                })
            });

            if (!response.ok) {
                const errorResult = await response.json();
                addMessageToLog({ type: 'error', content: `Server error: ${errorResult.detail || response.statusText}` }, thinkingProcessDiv);
                return;
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                const jsonObjects = chunk.split('\n').filter(str => str.trim() !== '');
                jsonObjects.forEach(jsonStr => {
                    try {
                        const data = JSON.parse(jsonStr);
                        handleStreamedData(data);
                    } catch (e) { console.error('Failed to parse streamed JSON:', jsonStr, e); }
                });
            }
        } catch (error) {
            console.error('Error sending goal:', error);
            addMessageToLog({ type: 'error', content: 'Failed to connect or stream response.' }, thinkingProcessDiv);
        }
    }

    function handleStreamedData(data) {
        const target = (data.type === 'final_answer' || data.type === 'user_goal') ? conversationDiv : thinkingProcessDiv;
        let prefix = '';
        if (data.type === 'final_answer') prefix = 'Agent Answer:';
        // else if (data.type === 'user_goal') prefix = 'Your Goal:'; // Already handled
        addMessageToLog(data, target, prefix);
    }

    function addMessageToLog(data, targetDiv, prefix = '') {
        const messageElement = document.createElement('div');
        messageElement.classList.add('log-message', `log-${data.type}`);
        let contentHTML = `<strong>${prefix || data.type.replace(/_/g, ' ').toUpperCase()}:</strong> `;
        
        if (data.type === 'prompt_context' && data.data) { // Handle nested data for prompt_context
            contentHTML += `<pre>${JSON.stringify(data.data, null, 2)}</pre>`;
        } else if (typeof data.content === 'object') {
            contentHTML += `<pre>${JSON.stringify(data.content, null, 2)}</pre>`;
        } else if (data.content !== undefined) {
            contentHTML += escapeHtml(String(data.content));
        }

        if (data.tool_name) contentHTML += `<br/>Tool: ${escapeHtml(data.tool_name)}`;
        if (data.tool_args) contentHTML += `<br/>Args: <pre>${escapeHtml(JSON.stringify(data.tool_args, null, 2))}</pre>`;
        if (data.iteration !== undefined) contentHTML += ` Iteration: ${data.iteration}/${data.max_iterations}`;
        
        messageElement.innerHTML = contentHTML;
        targetDiv.appendChild(messageElement);
        targetDiv.scrollTop = targetDiv.scrollHeight;
    }

    function escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) return '';
        return unsafe.toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    // --- Memory Controls ---
    searchMemoryBtn.addEventListener('click', async () => {
        const query = memorySearchInput.value.trim();
        if (!query) return;
        memorySearchResultsDiv.innerHTML = 'Searching...';
        try {
            const response = await fetch(`/api/memory/search?query=${encodeURIComponent(query)}&session_id=${currentSessionId}`);
            if (!response.ok) { const err = await response.json(); memorySearchResultsDiv.innerHTML = `Error: ${err.detail || response.statusText}`; return; }
            const results = await response.json();
            memorySearchResultsDiv.innerHTML = '';
            if (results.results && results.results.length > 0) {
                const ul = document.createElement('ul');
                results.results.forEach(item => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>Content:</strong> ${escapeHtml(item.content)} <br/> <strong>Metadata:</strong> <pre>${escapeHtml(JSON.stringify(item.metadata, null, 2))}</pre> <strong>Score:</strong> ${item.score.toFixed(4)}`;
                    ul.appendChild(li);
                });
                memorySearchResultsDiv.appendChild(ul);
            } else { memorySearchResultsDiv.textContent = 'No results found.'; }
        } catch (error) { console.error('Error searching memory:', error); memorySearchResultsDiv.textContent = 'Error performing memory search.'; }
    });

    clearMemoryBtn.addEventListener('click', async () => {
        if (!confirm(`Are you sure you want to clear memory for session ${currentSessionId}? This includes vector data if implemented.`)) return;
        memorySearchResultsDiv.innerHTML = 'Clearing memory...';
        try {
            const response = await fetch(`/api/memory/clear`, { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: currentSessionId})
            });
            if (!response.ok) { const err = await response.json(); memorySearchResultsDiv.innerHTML = `Error: ${err.detail || response.statusText}`; return; }
            const result = await response.json();
            memorySearchResultsDiv.textContent = result.message || 'Memory cleared.';
        } catch (error) { console.error('Error clearing memory:', error); memorySearchResultsDiv.textContent = 'Error clearing memory.'; }
    });

    // Initial load
    fetchAgentProfiles(); // This will also trigger selectAgentForSession and then fetchSessionLLMParameters
    // fetchSessionLLMParameters(); // Called by fetchAgentProfiles after selection
});