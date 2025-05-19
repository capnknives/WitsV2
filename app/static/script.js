document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const goalInput = document.getElementById('goal-input');
    const sendGoalBtn = document.getElementById('send-goal-btn');
    const conversationDiv = document.getElementById('conversation');
    const thinkingProcessDiv = document.getElementById('thinking-process');
    const loadingIndicator = document.getElementById('loading-indicator'); // Added loading indicator element
    const agentDescriptionBox = document.getElementById('agent-description-box'); // For agent description

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
    
    // Clear Chat & Log Buttons
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const clearThinkingBtn = document.getElementById('clear-thinking-btn');

    // New elements
    const uploadBtn = document.getElementById('upload-btn');
    const fileUpload = document.getElementById('file-upload');
    
    // Tab Navigation
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Show selected tab content
            tabContents.forEach(content => {
                content.style.display = 'none';
                if (content.id === `${tabName}-tab`) {
                    content.style.display = 'block';
                }
            });
            
            // Load data for status tab if selected
            if (tabName === 'status') {
                fetchSystemStatus();
            }
        });
    });    // Initialize session
    let currentSessionId = localStorage.getItem('witsNexusSessionId') || generateSessionId();
    localStorage.setItem('witsNexusSessionId', currentSessionId);
    sessionIdInput.value = currentSessionId;
    memorySessionIdDisplay.textContent = formatSessionId(currentSessionId);
    
    function generateSessionId() {
        return `web_session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
    }
    
    function formatSessionId(id) {
        return id.length > 15 ? `${id.substring(0,15)}...` : id;
    }

    // Function to set empty state for a log area
    function setEmptyState(logContainer, message, iconClass = 'fas fa-info-circle') {
        const messagesContainer = logContainer.querySelector('.messages-container');
        if (messagesContainer) {
            messagesContainer.innerHTML = `
                <div class=\"empty-state\">
                    <i class=\"${iconClass} fa-3x\"></i>
                    <p>${message}</p>
                </div>`;
        } else { // Fallback for older structure if .messages-container is not there (should not happen with new HTML)
            logContainer.innerHTML = `
                <div class=\"empty-state\">
                    <i class=\"${iconClass} fa-3x\"></i>
                    <p>${message}</p>
                </div>`;
        }
    }

    newSessionBtn.addEventListener('click', () => {
        currentSessionId = generateSessionId();
        localStorage.setItem('witsNexusSessionId', currentSessionId);
        sessionIdInput.value = currentSessionId;
        memorySessionIdDisplay.textContent = formatSessionId(currentSessionId);
        
        // Use setEmptyState for conversation and thinking logs
        setEmptyState(conversationDiv, 'Your conversation with WITS-NEXUS will appear here', 'fas fa-comments');
        
        const thinkingH3 = thinkingProcessDiv.querySelector('.thinking-header h3'); // Preserve header
        setEmptyState(thinkingProcessDiv, 'Agent\'s reasoning and planning will appear here', 'fas fa-lightbulb');
        if (thinkingH3 && !thinkingProcessDiv.querySelector('.thinking-header h3')) {
            thinkingProcessDiv.querySelector('.thinking-header').prepend(thinkingH3);
        }

        memorySearchResultsDiv.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history fa-2x"></i>
                <p>Search results will appear here</p>
            </div>`;
        
        addMessageToLog({type: 'info', content: `New session started: ${currentSessionId}`}, thinkingProcessDiv);
        fetchAgentProfiles(); // Re-fetch profiles to reset agent selection
        fetchSessionLLMParameters(); // Fetch params for the new session
    });

    // --- Clear Chat/Log Functionality ---
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the current chat conversation?')) {
                setEmptyState(conversationDiv, 'Chat cleared. Start a new conversation!', 'fas fa-comments');
                // Optionally, send a signal to the backend to clear server-side chat history for the session if needed
                // addMessageToLog({ type: 'info', content: 'Chat cleared by user.' }, thinkingProcessDiv); // Log action
            }
        });
    }

    if (clearThinkingBtn) {
        clearThinkingBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the agent thinking log?')) {
                const thinkingH3 = thinkingProcessDiv.querySelector('.thinking-header h3');
                setEmptyState(thinkingProcessDiv, 'Thinking log cleared.', 'fas fa-lightbulb');
                // Ensure the header (H3) is preserved if it was inside .thinking-header
                // The setEmptyState for thinkingProcessDiv might wipe out the header if not handled carefully.
                // The current HTML structure has H3 inside .thinking-header, which is outside .messages-container
                // So, it should be fine. If H3 was inside messages-container, it would need re-adding.
            }
        });
    }

    // --- Agent Selection ---
    async function fetchAgentProfiles() {
        try {
            const response = await fetch('/api/agents');
            if (!response.ok) throw new Error(`Failed to fetch agent profiles: ${response.statusText}`);
            const profiles = await response.json();
            
            agentSelect.innerHTML = ''; // Clear existing
            agentDescriptionBox.innerHTML = ''; // Clear description box

            let previouslySelectedAgent = localStorage.getItem(`selectedAgent_${currentSessionId}`);

            if (profiles.length === 0) {
                agentDescriptionBox.innerHTML = '<p class="empty-state-text">No agent profiles available.</p>';
                return;
            }

            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.name;
                option.textContent = profile.display_name || profile.name;
                option.title = profile.description || ''; // Keep title for tooltip
                // Store description in a data attribute for easy access
                option.dataset.description = profile.description || 'No description available for this agent.';
                agentSelect.appendChild(option);
            });

            if (previouslySelectedAgent && profiles.some(p => p.name === previouslySelectedAgent)) {
                agentSelect.value = previouslySelectedAgent;
            } else if (profiles.length > 0) {
                 previouslySelectedAgent = profiles[0].name; 
                 agentSelect.value = previouslySelectedAgent;
            }
            
            if (agentSelect.value) { 
                 const selectedOption = agentSelect.options[agentSelect.selectedIndex];
                 agentDescriptionBox.innerHTML = `<p>${escapeHtml(selectedOption.dataset.description)}</p>`;
                 await selectAgentForSession(agentSelect.value); 
            }


        } catch (error) {
            console.error('Error fetching agent profiles:', error);
            addMessageToLog({ type: 'error', content: 'Failed to load agent profiles.' }, thinkingProcessDiv);
            agentDescriptionBox.innerHTML = '<p class="error-text">Could not load agent descriptions.</p>';
        }
    }

    agentSelect.addEventListener('change', async () => {
        const selectedProfileName = agentSelect.value;
        if (selectedProfileName) {
            const selectedOption = agentSelect.options[agentSelect.selectedIndex];
            agentDescriptionBox.innerHTML = `<p>${escapeHtml(selectedOption.dataset.description)}</p>`;
            await selectAgentForSession(selectedProfileName);
        } else {
            agentDescriptionBox.innerHTML = ''; // Clear if no agent is selected
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
            // Update description box based on the newly selected agent from the dropdown
            const selectedOption = Array.from(agentSelect.options).find(opt => opt.value === profileName);
            if (selectedOption && selectedOption.dataset.description) {
                agentDescriptionBox.innerHTML = `<p>${escapeHtml(selectedOption.dataset.description)}</p>`;
            } else {
                 agentDescriptionBox.innerHTML = '<p>No description available for this agent.</p>'; // Fallback
            }
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
        // Allow Shift+Enter for new line, Enter to send
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default Enter behavior (new line in textarea)
            sendGoal();
        }
    });

    // Auto-resize textarea
    goalInput.addEventListener('input', () => {
        goalInput.style.height = 'auto'; // Reset height to shrink if text is deleted
        let newHeight = goalInput.scrollHeight;
        const maxHeight = parseInt(getComputedStyle(goalInput).maxHeight); // Get max-height from CSS
        if (newHeight > maxHeight) {
            newHeight = maxHeight;
            goalInput.style.overflowY = 'auto'; // Show scrollbar when max height is reached
        } else {
            goalInput.style.overflowY = 'hidden'; // Hide scrollbar if below max height
        }
        goalInput.style.height = `${newHeight}px`;
    });

    async function sendGoal() {
        const goal = goalInput.value.trim();
        if (!goal) return;

        if (loadingIndicator) loadingIndicator.style.display = 'flex'; // Show loading indicator

        addMessageToLog({ type: 'user_goal', content: goal }, conversationDiv);
        goalInput.value = '';
        // Reset textarea height after sending
        goalInput.style.height = 'auto';
        goalInput.style.overflowY = 'hidden';

        thinkingProcessDiv.innerHTML = '<h3><i class="fas fa-lightbulb"></i> Agent Thinking</h3>'; // Updated title

        const currentAgentProfile = agentSelect.value; // Get currently selected agent profile

        try {
            const response = await fetch('/api/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: goal,  // Use 'message' for better conversational UX
                    goal: goal,     // Keep 'goal' for backward compatibility
                    session_id: currentSessionId,
                    agent_profile_name: currentAgentProfile // Send the selected agent profile
                })
            });

            if (!response.ok) {
                const errorResult = await response.json();
                addMessageToLog({ type: 'error', content: `Server error: ${errorResult.detail || response.statusText}` }, thinkingProcessDiv);
                // if (loadingIndicator) loadingIndicator.style.display = 'none'; // Hide indicator - moved to finally
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
                    } catch (e) { 
                        console.error('Failed to parse streamed JSON:', jsonStr, e); 
                        // Optionally add an error message to UI if parsing fails mid-stream
                        // addMessageToLog({ type: 'error', content: 'Error processing stream data.' }, thinkingProcessDiv);
                    }
                });
            }
        } catch (error) {
            console.error('Error sending goal:', error);
            addMessageToLog({ type: 'error', content: 'Failed to connect or stream response.' }, thinkingProcessDiv);
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = 'none'; // Hide loading indicator in finally block
        }
    }

    function handleStreamedData(data) {
        const target = (data.type === 'final_answer' || data.type === 'user_goal') ? conversationDiv : thinkingProcessDiv;
        let prefix = '';
        if (data.type === 'final_answer') prefix = 'Agent Answer:';
        else if (data.type === 'user_goal') prefix = 'You:'; // Changed from 'USER_GOAL'
        addMessageToLog(data, target, prefix);
    }

    function addMessageToLog(data, targetDiv, prefix = '') {
        const messageElement = document.createElement('div');
        let styleType = data.type;
        let iconClass = '';
        let isCollapsible = false;
        // let initialSummary = ''; // Not directly used here, summary is generated dynamically

        // Determine styleType for CSS class and icon
        switch (data.type) {
            case 'user_goal':
                styleType = 'user';
                iconClass = 'fas fa-user-circle';
                prefix = prefix || 'You';
                break;
            case 'final_answer':
                styleType = 'agent';
                iconClass = 'fas fa-robot';
                prefix = prefix || 'Agent';
                break;
            case 'thought':
                styleType = 'thought';
                iconClass = 'fas fa-comment-dots';
                isCollapsible = true;
                break;
            case 'tool_input':
            case 'tool_output':
            case 'tool_call':
            case 'tool':
                styleType = 'tool';
                iconClass = 'fas fa-cogs';
                isCollapsible = true;
                break;
            case 'info':
                styleType = 'info';
                iconClass = 'fas fa-info-circle';
                break;
            case 'error':
                styleType = 'error';
                iconClass = 'fas fa-exclamation-circle';
                break;
            default:
                styleType = data.type;
                if (targetDiv === thinkingProcessDiv && data.type !== 'prompt_context') {
                    iconClass = 'fas fa-stream';
                }
                break;
        }

        messageElement.classList.add('log-message', styleType);
        if (isCollapsible) {
            messageElement.classList.add('collapsible');
        }

        let displayPrefixText = prefix;
        if (!displayPrefixText && data.type) {
            displayPrefixText = data.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        }

        let messageContentHTML = ''; // Changed to HTML to handle code snippets
        let fullContentForCollapsible = '';

        // Code snippet detection and formatting
        const codeBlockRegex = /```([a-zA-Z]*)\\n([\\s\\S]*?)```/g;
        let rawContent = '';

        if (data.type === 'prompt_context' && data.data) {
            rawContent = JSON.stringify(data.data, null, 2);
            // This is already typically wrapped in <pre> by existing logic, let\'s ensure it is
            messageContentHTML = `<pre class="code-snippet json">${escapeHtml(rawContent)}</pre>`;
        } else if (typeof data.content === 'object') {
            rawContent = JSON.stringify(data.content, null, 2);
            messageContentHTML = `<pre class="code-snippet json">${escapeHtml(rawContent)}</pre>`;
        } else if (data.content !== undefined && data.content !== null) {
            rawContent = String(data.content);
            if (codeBlockRegex.test(rawContent)) {
                messageContentHTML = rawContent.replace(codeBlockRegex, (match, lang, code) => {
                    const languageClass = lang ? `language-${escapeHtml(lang)}` : '';
                    // Add copy button inside the pre tag for better positioning relative to the code
                    return `<div class="code-block-wrapper">
                                <button class="copy-code-btn" title="Copy code"><i class="fas fa-copy"></i> Copy</button>
                                <pre><code class="code-snippet ${languageClass}">${escapeHtml(code.trim())}</code></pre>
                            </div>`;
                });
            } else {
                messageContentHTML = escapeHtml(rawContent);
            }
        } else if (data.type === 'user_goal' && data.goal) {
            rawContent = String(data.goal);
            messageContentHTML = escapeHtml(rawContent);
        }
        
        fullContentForCollapsible = messageContentHTML; // Use the potentially HTML formatted content

        let toolInfoHTML = '';
        if (data.tool_name) toolInfoHTML += `<div class="tool-info">Tool: ${escapeHtml(data.tool_name)}</div>`;
        if (data.tool_args) {
            const argsString = escapeHtml(JSON.stringify(data.tool_args, null, 2));
            toolInfoHTML += `<div class="tool-info">Args: <pre class="code-snippet json">${argsString}</pre></div>`;
        }
        if (data.iteration !== undefined && data.max_iterations !== undefined) {
            toolInfoHTML += `<div class="iteration-info">Iteration: ${data.iteration}/${data.max_iterations}</div>`;
        }

        if (isCollapsible && toolInfoHTML) {
            fullContentForCollapsible += toolInfoHTML;
            toolInfoHTML = '';
        }

        const timestampHTML = `<span class="timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>`;
        let iconContainerHTML = iconClass ? `<span class="msg-icon-container"><i class="${iconClass} msg-icon ${styleType}-icon"></i></span>` : '';
        let mainContentPrefixHTML = displayPrefixText ? `<strong>${escapeHtml(displayPrefixText.endsWith(':') || displayPrefixText.endsWith('?') || displayPrefixText.endsWith('!') ? displayPrefixText : displayPrefixText + ':')}</strong> ` : '';

        if (isCollapsible) {
            let summaryText;
            // Generate a text-only summary for the header
            let textOnlyContentForSummary = rawContent; 
            if (typeof data.content === 'object') textOnlyContentForSummary = JSON.stringify(data.content);


            if (data.type === 'tool' || data.type === 'tool_call' || data.type === 'tool_input' || data.type === 'tool_output') {
                summaryText = `Tool: ${escapeHtml(data.tool_name || 'Details')} ${data.tool_args ? '(click to expand args)' : '(click to expand)'}`;
            } else if (data.type === 'thought') {
                summaryText = `Thought: ${(textOnlyContentForSummary.length > 70 ? textOnlyContentForSummary.substring(0, 67) + "..." : textOnlyContentForSummary) || '(click to expand)'}`;
            } else {
                 summaryText = (textOnlyContentForSummary.length > 100 ? textOnlyContentForSummary.substring(0, 97) + "..." : textOnlyContentForSummary) || '(click to expand)';
            }


            messageElement.innerHTML = `
                ${iconContainerHTML}
                <div class="msg-content-container">
                    <div class="collapsible-header" role="button" tabindex="0" aria-expanded="false">
                        <span class="toggle-icon fas fa-chevron-right"></span>
                        ${mainContentPrefixHTML}
                        <span class="summary-text">${escapeHtml(summaryText)}</span>
                    </div>
                    <div class="collapsible-content">
                        ${fullContentForCollapsible} 
                        ${ data.type === 'tool' || data.type === 'tool_call' || data.type === 'tool_input' || data.type === 'tool_output' ? toolInfoHTML : ''}
                        ${timestampHTML}
                    </div>
                </div>
            `;
            
            const header = messageElement.querySelector('.collapsible-header');
            const content = messageElement.querySelector('.collapsible-content');
            const toggleIcon = messageElement.querySelector('.toggle-icon');

            header.addEventListener('click', () => {
                const isExpanded = content.classList.toggle('expanded');
                toggleIcon.classList.toggle('fa-chevron-right', !isExpanded);
                toggleIcon.classList.toggle('fa-chevron-down', isExpanded);
                header.setAttribute('aria-expanded', isExpanded);
            });
            header.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    header.click();
                }
            });

        } else {
            messageElement.innerHTML = `
                ${iconContainerHTML}
                <div class="msg-content-container">
                    <div class="main-content">${mainContentPrefixHTML}${messageContentHTML}</div>
                    ${toolInfoHTML}
                    ${timestampHTML}
                </div>
            `;
        }
        
        // Attach copy functionality to all new copy buttons
        messageElement.querySelectorAll('.copy-code-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                const wrapper = event.target.closest('.code-block-wrapper');
                const codeElement = wrapper ? wrapper.querySelector('pre code.code-snippet') : null;
                if (codeElement) {
                    copyToClipboard(codeElement.textContent, button);
                } else { // Fallback for tool_args pre blocks if needed, though they don\'t have a button yet
                    const preElement = event.target.closest('.tool-info')?.querySelector('pre.code-snippet');
                    if (preElement) {
                         copyToClipboard(preElement.textContent, button);
                    }
                }
            });
        });

        // Clear empty state message if it exists
        const messagesContainerConv = conversationDiv.querySelector('.messages-container');
        const emptyStateConversation = messagesContainerConv ? messagesContainerConv.querySelector('.empty-state') : conversationDiv.querySelector('.empty-state');
        if (emptyStateConversation && (styleType === 'user' || styleType === 'agent')) {
            if (messagesContainerConv) messagesContainerConv.innerHTML = '';
            else conversationDiv.innerHTML = '';
        }

        const messagesContainerThinking = thinkingProcessDiv.querySelector('.messages-container');
        const emptyStateThinking = messagesContainerThinking ? messagesContainerThinking.querySelector('.empty-state') : thinkingProcessDiv.querySelector('.empty-state');
        if (emptyStateThinking && targetDiv === thinkingProcessDiv) {
            if (messagesContainerThinking) {
                 if (messagesContainerThinking.querySelector('.empty-state')) messagesContainerThinking.innerHTML = '';
            } else {
                // Original logic if no .messages-container
                const h3 = thinkingProcessDiv.querySelector('h3');
                thinkingProcessDiv.innerHTML = '';
                if(h3) thinkingProcessDiv.appendChild(h3);
            }
        }
        
        // Append to correct container
        const targetMessagesContainer = targetDiv.querySelector('.messages-container');
        if (targetMessagesContainer) {
            targetMessagesContainer.appendChild(messageElement);
            targetMessagesContainer.scrollTop = targetMessagesContainer.scrollHeight;
        } else {
            targetDiv.appendChild(messageElement); // Fallback if no .messages-container
            targetDiv.scrollTop = targetDiv.scrollHeight;
        }
    }

    function escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) return '';
        // Added check for code snippets to prevent double escaping if already formatted
        if (typeof unsafe === 'string' && unsafe.includes('<pre class="code-snippet">')) {
            return unsafe;
        }
        return unsafe.toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    // --- Utility function to copy text to clipboard ---
    function copyToClipboard(text, buttonElement) {
        navigator.clipboard.writeText(text).then(() => {
            const originalButtonText = buttonElement.innerHTML;
            buttonElement.innerHTML = '<i class="fas fa-check"></i> Copied!';
            buttonElement.disabled = true;
            setTimeout(() => {
                buttonElement.innerHTML = originalButtonText;
                buttonElement.disabled = false;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            const originalButtonText = buttonElement.innerHTML;
            buttonElement.innerHTML = '<i class="fas fa-times"></i> Failed';
            setTimeout(() => {
                buttonElement.innerHTML = originalButtonText;
            }, 2000);
        });
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

    // File Upload Handling
    if (uploadBtn && fileUpload) {
        function handleFileUpload() {
            const file = fileUpload.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', currentSessionId);
            
            // Add file message to conversation
            addMessageToLog({
                type: 'info',
                content: `Uploading file: ${file.name}`
            }, conversationDiv);
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
                return response.json();
            })
            .then(data => {
                addMessageToLog({
                    type: 'info',
                    content: `File uploaded successfully: ${file.name}`
                }, conversationDiv);
                
                // If we got a file path back, add it to the goal input
                if (data.file_path) {
                    goalInput.value = `Analyze the uploaded file: ${data.file_path}`;
                }
            })
            .catch(error => {
                console.error('Error uploading file:', error);
                addMessageToLog({
                    type: 'error',
                    content: `File upload failed: ${error.message}`
                }, conversationDiv);
            });
        }
    }
    
    // System Status Updates
    function fetchSystemStatus() {
        fetch('/api/debug/metrics')
            .then(response => response.json())
            .then(data => {
                // Update status values
                document.getElementById('api-status').textContent = 'Online';
                document.getElementById('current-model').textContent = modelSelect.value || 'Not set';
                document.getElementById('active-sessions').textContent = data.active_sessions || '0';
                document.getElementById('memory-segments').textContent = data.memory_segments || '0';
                document.getElementById('avg-response-time').textContent = `${Math.round(data.avg_response_time || 0)} ms`;
                document.getElementById('llm-calls').textContent = data.llm_calls || '0';
                document.getElementById('tool-calls').textContent = data.tool_calls || '0';
            })
            .catch(error => {
                console.error('Error fetching system status:', error);
                document.getElementById('api-status').textContent = 'Offline';
                document.getElementById('api-status').className = 'status-value error';
            });
    }
    
    // Initial load
    fetchAgentProfiles(); // This will also trigger selectAgentForSession and then fetchSessionLLMParameters
    // fetchSessionLLMParameters(); // Called by fetchAgentProfiles after selection
});