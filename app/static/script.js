/**
 * @file script.js
 * @description Main JavaScript for WITS-NEXUS frontend, handling UI interactions,
 *              API calls, WebSocket communication, and dynamic content updates.
 */

/**
 * Initializes the application once the DOM is fully loaded.
 * Sets up event listeners, fetches initial data, and prepares the UI.
 */
document.addEventListener('DOMContentLoaded', () => {
    // DOM Element Cache
    /** @type {HTMLInputElement} */
    const goalInput = document.getElementById('goal-input');
    /** @type {HTMLButtonElement} */
    const sendGoalBtn = document.getElementById('send-goal-btn');
    /** @type {HTMLDivElement} */
    const conversationDiv = document.getElementById('conversation');
    /** @type {HTMLDivElement} */
    const thinkingProcessDiv = document.getElementById('thinking-process');
    /** @type {HTMLDivElement} */
    const loadingIndicator = document.getElementById('loading-indicator');
    /** @type {HTMLDivElement} */
    const agentDescriptionBox = document.getElementById('agent-description-box');
    /** @type {HTMLInputElement} */
    const sessionIdInput = document.getElementById('session-id-input');
    /** @type {HTMLButtonElement} */
    const newSessionBtn = document.getElementById('new-session-btn');
    /** @type {HTMLSelectElement} */
    const agentSelect = document.getElementById('agent-select');
    /** @type {HTMLSelectElement} */
    const modelSelect = document.getElementById('model-select');
    /** @type {HTMLInputElement} */
    const temperatureSlider = document.getElementById('temperature-slider');
    /** @type {HTMLSpanElement} */
    const temperatureValueSpan = document.getElementById('temperature-value');
    /** @type {HTMLButtonElement} */
    const updateParamsBtn = document.getElementById('update-params-btn');
    /** @type {HTMLInputElement} */
    const memorySearchInput = document.getElementById('memory-search-input');
    /** @type {HTMLButtonElement} */
    const searchMemoryBtn = document.getElementById('search-memory-btn');
    /** @type {HTMLButtonElement} */
    const clearMemoryBtn = document.getElementById('clear-memory-btn');
    /** @type {HTMLDivElement} */
    const memorySearchResultsDiv = document.getElementById('memory-search-results');
    /** @type {HTMLSpanElement} */
    const memorySessionIdDisplay = document.getElementById('memory-session-id-display');
    /** @type {HTMLButtonElement} */
    const clearChatBtn = document.getElementById('clear-chat-btn');
    /** @type {HTMLButtonElement} */
    const clearThinkingBtn = document.getElementById('clear-thinking-btn');
    /** @type {HTMLElement} */
    const plotOutlineSection = document.getElementById('plot-outline-section');
    /** @type {HTMLDivElement} */
    const plotOutlineContent = document.getElementById('plot-outline-content');
    /** @type {HTMLElement} */
    const characterProfilesSection = document.getElementById('character-profiles-section');
    /** @type {HTMLDivElement} */
    const characterProfilesContent = document.getElementById('character-profiles-content');
    /** @type {HTMLDivElement} */
    const worldDetailsContent = document.getElementById('world-details-content');
    /** @type {HTMLDivElement} */
    const globalErrorNotificationArea = document.getElementById('global-error-notification-area');
    /** @type {HTMLButtonElement} */
    const uploadBtn = document.getElementById('upload-btn'); // Assuming it's a button, adjust if not
    /** @type {HTMLInputElement} */
    const fileUpload = document.getElementById('file-upload');
    /** @type {NodeListOf<HTMLDivElement>} */
    const tabs = document.querySelectorAll('.tab');
    /** @type {NodeListOf<HTMLDivElement>} */
    const tabContents = document.querySelectorAll('.tab-content');

    // Session State
    /** @type {string} */
    let currentSessionId = '';

    // Initialization
    initializeSession();
    setupEventListeners();
    initializeUIStates();
    fetchInitialData();

    /**
     * Generates a unique session ID.
     * @returns {string} A unique identifier for the session.
     */
    function generateSessionId() {
        return `web_session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
    }

    /**
     * Formats a session ID for display, truncating if too long.
     * @param {string} id - The session ID to format.
     * @returns {string} The formatted session ID.
     */
    function formatSessionId(id) {
        return id.length > 15 ? `${id.substring(0,15)}...` : id;
    }
    
    /**
     * Initializes or retrieves the current session ID.
     * Updates UI elements with the session ID.
     */
    function initializeSession() {
        currentSessionId = localStorage.getItem('witsNexusSessionId') || generateSessionId();
        localStorage.setItem('witsNexusSessionId', currentSessionId);
        if (sessionIdInput) sessionIdInput.value = currentSessionId;
        if (memorySessionIdDisplay) memorySessionIdDisplay.textContent = formatSessionId(currentSessionId);
    }

    /**
     * Sets the empty/initial state for a given log container.
     * @param {HTMLDivElement} logContainer - The container element.
     * @param {string} message - The message to display.
     * @param {string} [iconClass='fas fa-info-circle'] - FontAwesome icon class.
     */
    function setEmptyState(logContainer, message, iconClass = 'fas fa-info-circle') {
        if (!logContainer) return;
        const messagesContainer = logContainer.querySelector('.messages-container');
        const emptyStateHTML = `
            <div class="empty-state">
                <i class="${iconClass} fa-3x"></i>
                <p>${message}</p>
            </div>`;
        
        if (messagesContainer) {
            messagesContainer.innerHTML = emptyStateHTML;
        } else { 
            logContainer.innerHTML = emptyStateHTML;
        }
    }

    /**
     * Initializes the empty states for various UI sections.
     */
    function initializeUIStates() {
        setEmptyState(conversationDiv, 'Your conversation with WITS-NEXUS will appear here', 'fas fa-comments');
        const thinkingH3 = thinkingProcessDiv ? thinkingProcessDiv.querySelector('.thinking-header h3') : null;
        setEmptyState(thinkingProcessDiv, "Agent's reasoning and planning will appear here", 'fas fa-lightbulb');
        if (thinkingH3 && thinkingProcessDiv && !thinkingProcessDiv.querySelector('.thinking-header h3')) {
            const thinkingHeader = thinkingProcessDiv.querySelector('.thinking-header');
            if (thinkingHeader) thinkingHeader.prepend(thinkingH3);
        }

        if (memorySearchResultsDiv) {
            memorySearchResultsDiv.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-history fa-2x"></i>
                    <p>Search results will appear here</p>
                </div>`;
        }
        setEmptyState(plotOutlineContent, 'Plot details from the Book Plotter agent will appear here.', 'fas fa-feather-alt');
        setEmptyState(characterProfilesContent, 'Character details from the Character Developer agent will appear here.', 'fas fa-user-edit');
        setEmptyState(worldDetailsContent, 'World details from the World Builder agent will appear here.', 'fas fa-globe');
    }

    /**
     * Fetches initial data required for the application, like agent profiles.
     */
    function fetchInitialData() {
        fetchAgentProfiles(); // This will also trigger selectAgentForSession and then fetchSessionLLMParameters
    }

    /**
     * Sets up all primary event listeners for UI elements.
     */
    function setupEventListeners() {
        if (newSessionBtn) newSessionBtn.addEventListener('click', handleNewSession);
        if (clearChatBtn) clearChatBtn.addEventListener('click', handleClearChat);
        if (clearThinkingBtn) clearThinkingBtn.addEventListener('click', handleClearThinkingLog);
        if (agentSelect) agentSelect.addEventListener('change', handleAgentSelectionChange);
        if (temperatureSlider) temperatureSlider.addEventListener('input', handleTemperatureChange);
        if (updateParamsBtn) updateParamsBtn.addEventListener('click', handleUpdateLLMParams);
        if (sendGoalBtn) sendGoalBtn.addEventListener('click', sendGoal);
        if (goalInput) {
            goalInput.addEventListener('keypress', handleGoalInputKeypress);
            goalInput.addEventListener('input', autoResizeGoalInput);
        }
        if (searchMemoryBtn) searchMemoryBtn.addEventListener('click', handleMemorySearch);
        if (clearMemoryBtn) clearMemoryBtn.addEventListener('click', handleClearMemory);
        if (uploadBtn && fileUpload) { // Ensure both exist
            // If uploadBtn is the trigger for a hidden file input
            uploadBtn.addEventListener('click', () => fileUpload.click());
            fileUpload.addEventListener('change', handleFileUpload);
        } else if (fileUpload) { // If only fileUpload exists and is directly interacted with
            fileUpload.addEventListener('change', handleFileUpload);
        }

        setupTabNavigation();
    }

    /**
     * Sets up tab navigation functionality.
     */
    function setupTabNavigation() {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;
                
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                tabContents.forEach(content => {
                    content.style.display = 'none';
                    if (content.id === `${tabName}-tab`) {
                        content.style.display = 'block';
                    }
                });
                
                if (tabName === 'status') {
                    fetchSystemStatus();
                }
            });
        });
    }

    /**
     * Handles the creation of a new session.
     */
    function handleNewSession() {
        currentSessionId = generateSessionId();
        localStorage.setItem('witsNexusSessionId', currentSessionId);
        if (sessionIdInput) sessionIdInput.value = currentSessionId;
        if (memorySessionIdDisplay) memorySessionIdDisplay.textContent = formatSessionId(currentSessionId);
        
        initializeUIStates(); // Reset UI states for the new session

        addMessageToLog({type: 'info', content: `New session started: ${currentSessionId}`}, thinkingProcessDiv);
        fetchAgentProfiles(); 
        // fetchSessionLLMParameters(); // Called by fetchAgentProfiles
    }

    /**
     * Handles clearing the chat conversation area.
     */
    function handleClearChat() {
        if (confirm('Are you sure you want to clear the current chat conversation?')) {
            setEmptyState(conversationDiv, 'Chat cleared. Start a new conversation!', 'fas fa-comments');
            // Optionally, add a log message or send a backend request
        }
    }

    /**
     * Handles clearing the agent thinking log.
     */
    function handleClearThinkingLog() {
        if (confirm('Are you sure you want to clear the agent thinking log?')) {
            const thinkingH3 = thinkingProcessDiv ? thinkingProcessDiv.querySelector('.thinking-header h3') : null;
            setEmptyState(thinkingProcessDiv, 'Thinking log cleared.', 'fas fa-lightbulb');
            if (thinkingH3 && thinkingProcessDiv && !thinkingProcessDiv.querySelector('.thinking-header h3')) {
                 const thinkingHeader = thinkingProcessDiv.querySelector('.thinking-header');
                 if(thinkingHeader) thinkingHeader.prepend(thinkingH3);
            }
        }
    }

    /**
     * Displays a global error message in the notification area.
     * @param {string} message - The error message to display.
     * @param {boolean} [isHtml=false] - Whether the message is HTML or plain text.
     */
    function displayGlobalError(message, isHtml = false) {
        if (!globalErrorNotificationArea) return;

        const errorElement = document.createElement('div');
        errorElement.classList.add('global-error-message');
        
        if (isHtml) {
            errorElement.innerHTML = message; 
        } else {
            errorElement.textContent = message; 
        }

        const closeButton = document.createElement('button');
        closeButton.classList.add('close-error-btn');
        closeButton.innerHTML = '&times;'; 
        closeButton.setAttribute('aria-label', 'Close error message');
        closeButton.onclick = () => {
            errorElement.remove();
            if (globalErrorNotificationArea && !globalErrorNotificationArea.hasChildNodes()) {
                globalErrorNotificationArea.style.display = 'none'; 
            }
        };
        errorElement.appendChild(closeButton);

        globalErrorNotificationArea.appendChild(errorElement);
        globalErrorNotificationArea.style.display = 'block'; 
    }

    // --- Agent Management ---
    /**
     * Fetches agent profiles from the backend and populates the agent selection dropdown.
     * Updates the agent description box based on the selected or default agent.
     */
    async function fetchAgentProfiles() {
        if (!agentSelect || !agentDescriptionBox) return;
        try {
            const response = await fetch('/api/agents');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
            const profiles = await response.json();
            
            agentSelect.innerHTML = ''; 
            agentDescriptionBox.innerHTML = ''; 

            let previouslySelectedAgent = localStorage.getItem(`selectedAgent_${currentSessionId}`);

            if (!profiles || profiles.length === 0) {
                agentDescriptionBox.innerHTML = '<p class="empty-state-text">No agent profiles available.</p>';
                return;
            }

            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.name;
                option.textContent = profile.display_name || profile.name;
                option.title = profile.description || ''; 
                option.dataset.description = profile.description || 'No description available for this agent.';
                agentSelect.appendChild(option);
            });

            if (previouslySelectedAgent && profiles.some(p => p.name === previouslySelectedAgent)) {
                agentSelect.value = previouslySelectedAgent;
            } else if (profiles.length > 0) {
                 agentSelect.value = profiles[0].name; 
            }
            
            await updateAgentSelectionInUIAndBackend();

        } catch (error) {
            console.error('Error fetching agent profiles:', error);
            displayGlobalError(`Failed to load agent profiles: ${error.message}`);
            if (agentDescriptionBox) agentDescriptionBox.innerHTML = '<p class="error-text">Could not load agent descriptions.</p>';
        }
    }

    /**
     * Handles the change event of the agent selection dropdown.
     * Updates the agent description and notifies the backend.
     */
    async function handleAgentSelectionChange() {
        await updateAgentSelectionInUIAndBackend();
    }

    /**
     * Updates the agent description box and sends the selected agent to the backend.
     * This is a helper function for fetchAgentProfiles and handleAgentSelectionChange.
     */
    async function updateAgentSelectionInUIAndBackend() {
        if (!agentSelect || !agentDescriptionBox) return;
        const selectedProfileName = agentSelect.value;

        if (selectedProfileName) {
            const selectedOption = agentSelect.options[agentSelect.selectedIndex];
            if (selectedOption && selectedOption.dataset.description) {
                agentDescriptionBox.innerHTML = `<p>${escapeHtml(selectedOption.dataset.description)}</p>`;
            }
            await selectAgentForSession(selectedProfileName);
        } else {
            agentDescriptionBox.innerHTML = ''; 
        }
    }

    /**
     * Sends the selected agent profile to the backend for the current session.
     * @param {string} profileName - The name of the agent profile to select.
     */
    async function selectAgentForSession(profileName) {
        try {
            const response = await fetch('/api/session/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, agent_profile_name: profileName })
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            localStorage.setItem(`selectedAgent_${currentSessionId}`, profileName); 
            addMessageToLog({ type: 'info', content: `Switched to agent: ${profileName}. ${result.message || ''}` }, thinkingProcessDiv);
            
            // Ensure description box is updated if it wasn't by the caller
            const selectedOption = Array.from(agentSelect.options).find(opt => opt.value === profileName);
            if (selectedOption && selectedOption.dataset.description && agentDescriptionBox) {
                agentDescriptionBox.innerHTML = `<p>${escapeHtml(selectedOption.dataset.description)}</p>`;
            } else if (agentDescriptionBox) {
                 agentDescriptionBox.innerHTML = '<p>No description available for this agent.</p>';
            }
            fetchSessionLLMParameters(); 
        } catch (error) {
            console.error('Error selecting agent:', error);
            displayGlobalError(`Error switching agent: ${error.message}`);
        }
    }

    // --- LLM Parameter Management ---
    /**
     * Handles changes to the temperature slider.
     */
    function handleTemperatureChange() {
        if (temperatureValueSpan && temperatureSlider) {
            temperatureValueSpan.textContent = temperatureSlider.value;
        }
    }

    /**
     * Fetches LLM parameters for the current session and agent from the backend.
     * Updates the model selection dropdown and temperature slider.
     */
    async function fetchSessionLLMParameters() {
        if (!modelSelect || !temperatureSlider || !temperatureValueSpan) return;
        try {
            const response = await fetch(`/api/session/parameters?session_id=${currentSessionId}`);
            if (!response.ok) {
                displayGlobalError(`Failed to load session LLM parameters: ${response.statusText}`);
                return;
            }
            const params = await response.json();
            temperatureSlider.value = params.temperature || 0.7;
            temperatureValueSpan.textContent = temperatureSlider.value;

            modelSelect.innerHTML = ''; 
            if (params.available_models && params.available_models.length > 0) {
                params.available_models.forEach(modelName => {
                    const option = document.createElement('option');
                    option.value = modelName;
                    option.textContent = modelName;
                    if (modelName === params.model) {
                        option.selected = true;
                    }
                    modelSelect.appendChild(option);
                });
            } else {
                const option = document.createElement('option');
                option.textContent = 'No models available';
                option.disabled = true;
                modelSelect.appendChild(option);
            }

        } catch (error) {
            console.error('Error fetching session LLM parameters:', error);
            displayGlobalError(`Error loading session LLM parameters: ${error.message}`);
        }
    }

    /**
     * Handles the click event for updating LLM parameters.
     * Sends the new parameters to the backend.
     */
    async function handleUpdateLLMParams() {
        if (!modelSelect || !temperatureSlider) return;
        const payload = {
            session_id: currentSessionId,
            model: modelSelect.value,
            temperature: parseFloat(temperatureSlider.value)
        };
        addMessageToLog({type:'info', content: `Updating LLM params for agent ${agentSelect.options[agentSelect.selectedIndex].text}...`}, thinkingProcessDiv);
        try {
            const response = await fetch('/api/config/parameters', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok) {
                displayGlobalError(`Failed to update LLM parameters: ${result.detail || 'Unknown error'}`);
            } else {
                addMessageToLog({ type: 'info', content: result.message || 'Parameters updated.' }, thinkingProcessDiv);
            }
        } catch (error) {
            console.error('Error updating LLM parameters:', error);
            displayGlobalError(`Error updating LLM parameters: ${error.message}`);
        }
    }

    // --- Chat & Goal Submission ---
    /**
     * Handles keypress events in the goal input field (Enter to send).
     * @param {KeyboardEvent} e - The keyboard event.
     */
    function handleGoalInputKeypress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            sendGoal();
        }
    }

    /**
     * Automatically resizes the goal input textarea based on its content.
     */
    function autoResizeGoalInput() {
        if (!goalInput) return;
        goalInput.style.height = 'auto'; 
        let newHeight = goalInput.scrollHeight;
        const maxHeight = parseInt(getComputedStyle(goalInput).maxHeight); 
        if (newHeight > maxHeight) {
            newHeight = maxHeight;
            goalInput.style.overflowY = 'auto'; 
        } else {
            goalInput.style.overflowY = 'hidden'; 
        }
        goalInput.style.height = `${newHeight}px`;
    }

    /**
     * Sends the user's goal/message to the backend via a streaming API.
     * Handles the streamed response and updates UI accordingly.
     */
    async function sendGoal() {
        if (!goalInput || !agentSelect) return;
        const goal = goalInput.value.trim();
        if (!goal) return;

        if (loadingIndicator) loadingIndicator.style.display = 'flex'; 

        addMessageToLog({ type: 'user_goal', content: goal }, conversationDiv);
        goalInput.value = '';
        autoResizeGoalInput(); // Reset height

        if (thinkingProcessDiv) {
            const thinkingHeader = thinkingProcessDiv.querySelector('.thinking-header');
            const messagesContainer = thinkingProcessDiv.querySelector('.messages-container');
            if (thinkingHeader && messagesContainer) {
                messagesContainer.innerHTML = ''; // Clear previous thinking messages
            } else if (thinkingProcessDiv) { // Fallback if structure is different
                thinkingProcessDiv.innerHTML = '<h3><i class="fas fa-lightbulb"></i> Agent Thinking</h3>'; 
            }
        }

        const currentAgentProfile = agentSelect.value; 

        try {
            const response = await fetch('/api/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: goal,  
                    goal: goal,     
                    session_id: currentSessionId,
                    agent_profile_name: currentAgentProfile 
                })
            });

            if (!response.ok) {
                const errorResult = await response.json();
                displayGlobalError(`Chat stream error: ${errorResult.detail || response.statusText}`);
                return;
            }
            // Stream processing logic (remains largely the same, but ensure it calls handleStreamedData)
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
                        displayGlobalError('Error processing streamed data. Check console for details.');
                    }
                });
            }
        } catch (error) {
            console.error('Error sending goal:', error);
            displayGlobalError(`Failed to send message or stream response: ${error.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = 'none'; 
        }
    }

    /**
     * Handles incoming streamed data from the backend.
     * Routes data to the appropriate display area (conversation, thinking log, book sections).
     * @param {object} data - The streamed data object.
     */
    function handleStreamedData(data) {
        // Determine target and prefix (simplified)
        const isConversationMessage = data.type === 'final_answer' || data.type === 'user_goal';
        const targetDiv = isConversationMessage ? conversationDiv : thinkingProcessDiv;
        let prefix = data.type === 'final_answer' ? 'Agent Answer:' : (data.type === 'user_goal' ? 'You:' : '');

        // Specific handling for book writing agents
        if (data.agent_name === 'book_plotter' && data.type === 'final_answer') {
            updateBookWritingOutput(plotOutlineSection, plotOutlineContent, data.content, 'plot-detail');
        } else if (data.agent_name === 'character_agent' && data.type === 'final_answer') {
            updateBookWritingOutput(characterProfilesSection, characterProfilesContent, data.content, 'character-profile-detail');
        } else if (data.agent_name === 'book_worldbuilder' && data.type === 'final_answer') {
            // Assuming data.data for worldbuilder, adjust if it uses data.content
            updateBookWritingOutput(null, worldDetailsContent, data.data, 'world-detail', true);
        } else if (['agent_log', 'tool_log', 'tool_input', 'tool_output', 'final_answer', 'error', 'info', 'debug', 'thought', 'tool_call'].includes(data.type)) {
            // Ensure all relevant types are passed to addMessageToLog
            addMessageToLog(data, targetDiv, prefix);
        }
    }

    /**
     * Helper function to update book writing output areas.
     * @param {HTMLElement | null} sectionElement - The main section element (e.g., plotOutlineSection). Can be null if not applicable.
     * @param {HTMLDivElement} contentElement - The content div element (e.g., plotOutlineContent).
     * @param {string} htmlContent - The HTML content to display.
     * @param {string} detailClass - The class to add to the detail element.
     * @param {boolean} [isRawHtml=false] - Whether the content is raw HTML or needs escaping.
     */
    function updateBookWritingOutput(sectionElement, contentElement, htmlContent, detailClass, isRawHtml = false) {
        if (!contentElement) return;
        if (sectionElement) sectionElement.style.display = 'block';
        
        // Clear empty state or initial message
        const emptyState = contentElement.querySelector('.empty-state');
        if (emptyState) {
            contentElement.innerHTML = ''; 
        } else if (contentElement.innerHTML.includes('<p>') && contentElement.firstChild && contentElement.firstChild.nodeName === 'P') {
            // More robust check for initial placeholder paragraph
            if (contentElement.textContent.includes('will appear here')) {
                 contentElement.innerHTML = '';
            }
        }

        const detailElement = document.createElement('div');
        detailElement.classList.add(detailClass);
        detailElement.innerHTML = isRawHtml ? htmlContent : escapeHtml(htmlContent);
        contentElement.appendChild(detailElement);
        contentElement.scrollTop = contentElement.scrollHeight;
    }

    /**
     * Adds a message to the specified log display area (conversation or thinking process).
     * Handles different message types, formatting, and collapsible sections.
     * @param {object} data - The message data object.
     * @param {HTMLDivElement} targetDiv - The DOM element to append the message to.
     * @param {string} [prefix=''] - A prefix for the message (e.g., 'Agent:').
     */
    function addMessageToLog(data, targetDiv, prefix = '') {
        if (!targetDiv) return;

        const messageElement = document.createElement('div');
        const { styleType, iconClass, isCollapsible } = getMessageStyleInfo(data, targetDiv);

        messageElement.classList.add('log-message', styleType);
        if (isCollapsible) {
            messageElement.classList.add('collapsible');
        }

        const displayPrefixText = getDisplayPrefix(data, prefix);
        const { messageContentHTML, rawContent } = formatMessageContent(data);
        let toolInfoHTML = formatToolInfo(data);

        let fullContentForCollapsible = messageContentHTML;
        if (isCollapsible && toolInfoHTML && (data.type === 'tool' || data.type === 'tool_call' || data.type === 'tool_input' || data.type === 'tool_output')) {
            // Append toolInfo to fullContentForCollapsible only for tool-related collapsible messages
            // For other collapsible types like 'thought', toolInfoHTML is usually empty or not relevant here.
            fullContentForCollapsible += toolInfoHTML;
            toolInfoHTML = ''; // Clear it as it's now part of fullContentForCollapsible
        }

        const timestampHTML = `<span class="timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>`;
        const iconContainerHTML = iconClass ? `<span class="msg-icon-container"><i class="${iconClass} msg-icon ${styleType}-icon"></i></span>` : '';
        const mainContentPrefixHTML = displayPrefixText ? `<strong>${escapeHtml(displayPrefixText.endsWith(':') || displayPrefixText.endsWith('?') || displayPrefixText.endsWith('!') ? displayPrefixText : displayPrefixText + ':')}</strong> ` : '';

        if (isCollapsible) {
            const summaryText = generateCollapsibleSummary(data, rawContent);
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
                        ${timestampHTML}
                    </div>
                </div>
            `;
            setupCollapsibleBehavior(messageElement);
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
        
        attachCopyButtonListeners(messageElement);
        clearEmptyStateForTarget(targetDiv, styleType);
        appendMessageToDOM(targetDiv, messageElement);
    }

    /**
     * Determines styling information for a log message based on its data.
     * @param {object} data - The message data.
     * @param {HTMLDivElement} targetDiv - The target display area.
     * @returns {{styleType: string, iconClass: string, isCollapsible: boolean}}
     */
    function getMessageStyleInfo(data, targetDiv) {
        let styleType = data.type;
        let iconClass = '';
        let isCollapsible = false;

        switch (data.type) {
            case 'user_goal':
                styleType = 'user';
                iconClass = 'fas fa-user-circle';
                break;
            case 'final_answer':
                styleType = 'agent';
                iconClass = 'fas fa-robot';
                break;
            case 'thought':
                styleType = 'thought';
                iconClass = 'fas fa-comment-dots';
                isCollapsible = true;
                break;
            case 'tool_input':
            case 'tool_output':
            case 'tool_call':
            case 'tool': // Generic tool log
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
                // styleType remains data.type
                if (targetDiv === thinkingProcessDiv && data.type !== 'prompt_context') {
                    iconClass = 'fas fa-stream'; // Default for other thinking process messages
                }
                break;
        }
        return { styleType, iconClass, isCollapsible };
    }

    /**
     * Generates the display prefix for a log message.
     * @param {object} data - The message data.
     * @param {string} initialPrefix - The initial prefix passed to addMessageToLog.
     * @returns {string} The display prefix string.
     */
    function getDisplayPrefix(data, initialPrefix) {
        if (initialPrefix) return initialPrefix;
        if (data.type) {
            // Capitalize first letter of each word, replace underscores with spaces
            return data.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
        return '';
    }

    /**
     * Formats the main content of a log message, handling code blocks.
     * @param {object} data - The message data.
     * @returns {{messageContentHTML: string, rawContent: string}}
     */
    function formatMessageContent(data) {
        const codeBlockRegex = /```([a-zA-Z]*)\n([\s\S]*?)```/g;
        let rawContent = '';
        let messageContentHTML = '';

        if (data.type === 'prompt_context' && data.data) {
            rawContent = JSON.stringify(data.data, null, 2);
            messageContentHTML = `<pre class="code-snippet json">${escapeHtml(rawContent)}</pre>`;
        } else if (typeof data.content === 'object') {
            rawContent = JSON.stringify(data.content, null, 2);
            messageContentHTML = `<pre class="code-snippet json">${escapeHtml(rawContent)}</pre>`;
        } else if (data.content !== undefined && data.content !== null) {
            rawContent = String(data.content);
            if (codeBlockRegex.test(rawContent)) {
                messageContentHTML = rawContent.replace(codeBlockRegex, (match, lang, code) => {
                    const languageClass = lang ? `language-${escapeHtml(lang)}` : '';
                    return `<div class="code-block-wrapper">
                                <button class="copy-code-btn" title="Copy code"><i class="fas fa-copy"></i> Copy</button>
                                <pre><code class="code-snippet ${languageClass}">${escapeHtml(code.trim())}</code></pre>
                            </div>`;
                });
            } else {
                messageContentHTML = escapeHtml(rawContent);
            }
        } else if (data.type === 'user_goal' && data.goal) { // Handle user_goal if content is in data.goal
            rawContent = String(data.goal);
            messageContentHTML = escapeHtml(rawContent);
        }
        return { messageContentHTML, rawContent };
    }

    /**
     * Formats tool-specific information for a log message.
     * @param {object} data - The message data.
     * @returns {string} HTML string for tool information.
     */
    function formatToolInfo(data) {
        let toolInfoHTML = '';
        if (data.tool_name) toolInfoHTML += `<div class="tool-info">Tool: ${escapeHtml(data.tool_name)}</div>`;
        if (data.tool_args) {
            const argsString = escapeHtml(JSON.stringify(data.tool_args, null, 2));
            toolInfoHTML += `<div class="tool-info">Args: <pre class="code-snippet json">${argsString}</pre></div>`;
        }
        if (data.iteration !== undefined && data.max_iterations !== undefined) {
            toolInfoHTML += `<div class="iteration-info">Iteration: ${data.iteration}/${data.max_iterations}</div>`;
        }
        return toolInfoHTML;
    }

    /**
     * Generates a summary text for collapsible log messages.
     * @param {object} data - The message data.
     * @param {string} rawContent - The raw text content of the message.
     * @returns {string} The summary text.
     */
    function generateCollapsibleSummary(data, rawContent) {
        let textOnlyContentForSummary = rawContent;
        if (typeof data.content === 'object') textOnlyContentForSummary = JSON.stringify(data.content);

        if (data.type === 'tool' || data.type === 'tool_call' || data.type === 'tool_input' || data.type === 'tool_output') {
            return `Tool: ${escapeHtml(data.tool_name || 'Details')} ${data.tool_args ? '(click to expand args)' : '(click to expand)'}`;
        } else if (data.type === 'thought') {
            return `Thought: ${(textOnlyContentForSummary.length > 70 ? textOnlyContentForSummary.substring(0, 67) + "..." : textOnlyContentForSummary) || '(click to expand)'}`;
        }
        return (textOnlyContentForSummary.length > 100 ? textOnlyContentForSummary.substring(0, 97) + "..." : textOnlyContentForSummary) || '(click to expand)';
    }

    /**
     * Sets up the behavior for a collapsible log message.
     * @param {HTMLDivElement} messageElement - The message element.
     */
    function setupCollapsibleBehavior(messageElement) {
        const header = messageElement.querySelector('.collapsible-header');
        const content = messageElement.querySelector('.collapsible-content');
        const toggleIcon = messageElement.querySelector('.toggle-icon');

        if (header && content && toggleIcon) {
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
        }
    }

    /**
     * Attaches event listeners to copy buttons within a message element.
     * @param {HTMLDivElement} messageElement - The message element.
     */
    function attachCopyButtonListeners(messageElement) {
        messageElement.querySelectorAll('.copy-code-btn').forEach(button => {
            button.addEventListener('click', (event) => {
                const wrapper = event.target.closest('.code-block-wrapper');
                const codeElement = wrapper ? wrapper.querySelector('pre code.code-snippet') : null;
                if (codeElement) {
                    copyToClipboard(codeElement.textContent, button);
                } else {
                    const preElement = event.target.closest('.tool-info')?.querySelector('pre.code-snippet');
                    if (preElement) {
                         copyToClipboard(preElement.textContent, button);
                    }
                }
            });
        });
    }

    /**
     * Clears the empty state message from a target display area if necessary.
     * @param {HTMLDivElement} targetDiv - The target display area.
     * @param {string} messageStyleType - The style type of the message being added.
     */
    function clearEmptyStateForTarget(targetDiv, messageStyleType) {
        const messagesContainer = targetDiv.querySelector('.messages-container');
        const emptyState = messagesContainer ? messagesContainer.querySelector('.empty-state') : targetDiv.querySelector('.empty-state');
        
        if (emptyState) {
            if (targetDiv === conversationDiv && (messageStyleType === 'user' || messageStyleType === 'agent')) {
                if (messagesContainer) messagesContainer.innerHTML = '';
                else targetDiv.innerHTML = '';
            } else if (targetDiv === thinkingProcessDiv) {
                if (messagesContainer) {
                     if (messagesContainer.querySelector('.empty-state')) messagesContainer.innerHTML = '';
                } else {
                    const h3 = targetDiv.querySelector('h3'); // Preserve header if outside messagesContainer
                    targetDiv.innerHTML = '';
                    if(h3) targetDiv.appendChild(h3);
                }
            }
        }
    }

    /**
     * Appends a new message element to the target display area in the DOM.
     * @param {HTMLDivElement} targetDiv - The target display area.
     * @param {HTMLDivElement} messageElement - The message element to append.
     */
    function appendMessageToDOM(targetDiv, messageElement) {
        const targetMessagesContainer = targetDiv.querySelector('.messages-container');
        if (targetMessagesContainer) {
            targetMessagesContainer.appendChild(messageElement);
            targetMessagesContainer.scrollTop = targetMessagesContainer.scrollHeight;
        } else {
            targetDiv.appendChild(messageElement); 
            targetDiv.scrollTop = targetDiv.scrollHeight;
        }
    }

    /**
     * Escapes HTML special characters in a string.
     * @param {*} unsafe - The string or value to escape.
     * @returns {string} The escaped string.
     */
    function escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) return '';
        // Avoid double escaping if content is already HTML (e.g. pre-formatted code blocks from server)
        if (typeof unsafe === 'string' && (unsafe.includes('<pre class="code-snippet">') || unsafe.includes('<div class="code-block-wrapper">'))) {
            return unsafe;
        }
        return unsafe.toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // --- Clipboard Utility ---
    /**
     * Copies text to the clipboard and provides user feedback on the button.
     * @param {string} text - The text to copy.
     * @param {HTMLButtonElement} buttonElement - The button that triggered the copy action.
     */
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
            buttonElement.disabled = true; // Also disable on fail for a bit
            setTimeout(() => {
                buttonElement.innerHTML = originalButtonText;
                buttonElement.disabled = false;
            }, 2000);
        });
    }

    // --- Memory Management ---
    /**
     * Handles the memory search functionality.
     */
    async function handleMemorySearch() {
        if (!memorySearchInput || !memorySearchResultsDiv) return;
        const query = memorySearchInput.value.trim();
        if (!query) return;

        memorySearchResultsDiv.innerHTML = '<div class="loading-state"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
        try {
            const response = await fetch(`/api/memory/search?query=${encodeURIComponent(query)}&session_id=${currentSessionId}`);
            if (!response.ok) { 
                const err = await response.json(); 
                displayGlobalError(`Memory search error: ${err.detail || response.statusText}`);
                memorySearchResultsDiv.innerHTML = '<p class="error-text">Search failed. See global errors.</p>';
                return; 
            }
            const results = await response.json();
            renderMemorySearchResults(results);
        } catch (error) { 
            console.error('Error searching memory:', error); 
            displayGlobalError(`Error performing memory search: ${error.message}`);
            if (memorySearchResultsDiv) memorySearchResultsDiv.innerHTML = '<p class="error-text">Search failed due to a network or parsing error.</p>';
        }
    }

    /**
     * Renders the memory search results in the UI.
     * @param {object} results - The search results from the API.
     */
    function renderMemorySearchResults(results) {
        if (!memorySearchResultsDiv) return;
        memorySearchResultsDiv.innerHTML = '';
        if (results.results && results.results.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'memory-results-list';
            results.results.forEach(item => {
                const li = document.createElement('li');
                li.className = 'memory-result-item';
                li.innerHTML = `
                    <div class="result-content"><strong>Content:</strong> ${escapeHtml(item.content)}</div>
                    <div class="result-metadata"><strong>Metadata:</strong> <pre>${escapeHtml(JSON.stringify(item.metadata, null, 2))}</pre></div>
                    <div class="result-score"><strong>Score:</strong> ${item.score.toFixed(4)}</div>`;
                ul.appendChild(li);
            });
            memorySearchResultsDiv.appendChild(ul);
        } else { 
            setEmptyState(memorySearchResultsDiv, 'No results found for your query.', 'fas fa-search');
        }
    }

    /**
     * Handles clearing the memory for the current session.
     */
    async function handleClearMemory() {
        if (!memorySearchResultsDiv) return;
        if (!confirm(`Are you sure you want to clear all memory for session ${formatSessionId(currentSessionId)}? This action cannot be undone.`)) return;
        
        memorySearchResultsDiv.innerHTML = '<div class="loading-state"><i class="fas fa-spinner fa-spin"></i> Clearing memory...</div>';
        try {
            const response = await fetch(`/api/memory/clear`, { 
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: currentSessionId})
            });
            if (!response.ok) { 
                const err = await response.json(); 
                displayGlobalError(`Failed to clear memory: ${err.detail || response.statusText}`);
                memorySearchResultsDiv.innerHTML = '<p class="error-text">Clear memory failed. See global errors.</p>';
                return; 
            }
            const result = await response.json();
            setEmptyState(memorySearchResultsDiv, result.message || 'Memory cleared successfully.', 'fas fa-check-circle');
            addMessageToLog({type: 'info', content: 'Memory cleared for current session.'}, thinkingProcessDiv);
        } catch (error) { 
            console.error('Error clearing memory:', error); 
            displayGlobalError(`Error clearing memory: ${error.message}`);
            if (memorySearchResultsDiv) memorySearchResultsDiv.innerHTML = '<p class="error-text">Clear memory failed due to a network or parsing error.</p>';
        }
    }

    // --- File Upload Management ---
    /**
     * Handles the file upload process.
     */
    async function handleFileUpload() {
        if (!fileUpload || !fileUpload.files || fileUpload.files.length === 0) return;
        const file = fileUpload.files[0];
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', currentSessionId);
        
        addMessageToLog({ type: 'info', content: `Uploading file: ${file.name}...` }, conversationDiv);
        if (loadingIndicator) loadingIndicator.style.display = 'flex';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: `Upload failed with status: ${response.status}` }));
                throw new Error(errData.detail || `Upload failed: ${response.statusText}`);
            }
            const data = await response.json();
            addMessageToLog({ type: 'info', content: `File uploaded successfully: ${file.name}. Path: ${data.file_path}` }, conversationDiv);
            
            if (goalInput && data.file_path) {
                goalInput.value = `${goalInput.value} Context from uploaded file: ${data.file_path}`.trim();
                autoResizeGoalInput();
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            displayGlobalError(`File upload failed: ${error.message}`);
            addMessageToLog({ type: 'error', content: `Failed to upload ${file.name}: ${error.message}` }, conversationDiv);
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = 'none';
            if (fileUpload) fileUpload.value = ''; // Reset file input
        }
    }
    
    // --- System Status Updates ---
    /**
     * Fetches and displays system status and metrics.
     */
    function fetchSystemStatus() {
        fetch('/api/debug/metrics')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                updateStatusDisplay('api-status', 'Online', data.error ? 'error' : 'success');
                updateStatusDisplay('current-model', modelSelect && modelSelect.value ? modelSelect.value : 'Not set');
                updateStatusDisplay('active-sessions', data.active_sessions !== undefined ? String(data.active_sessions) : 'N/A');
                updateStatusDisplay('memory-segments', data.memory_segments !== undefined ? String(data.memory_segments) : 'N/A');
                updateStatusDisplay('avg-response-time', data.avg_response_time !== undefined ? `${Math.round(data.avg_response_time)} ms` : 'N/A');
                updateStatusDisplay('llm-calls', data.llm_calls !== undefined ? String(data.llm_calls) : 'N/A');
                updateStatusDisplay('tool-calls', data.tool_calls !== undefined ? String(data.tool_calls) : 'N/A');
            })
            .catch(error => {
                console.error('Error fetching system status:', error);
                updateStatusDisplay('api-status', 'Offline', 'error');
                ['current-model', 'active-sessions', 'memory-segments', 'avg-response-time', 'llm-calls', 'tool-calls']
                    .forEach(id => updateStatusDisplay(id, 'Error', 'error'));
                displayGlobalError('Failed to fetch system status. API might be offline or an error occurred.');
            });
    }

    /**
     * Helper to update a status display element.
     * @param {string} id - The ID of the status element.
     * @param {string} text - The text to display.
     * @param {string} [statusClass] - Optional class ('success', 'error') to add.
     */
    function updateStatusDisplay(id, text, statusClass) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = text;
            element.classList.remove('success', 'error'); // Remove old status classes
            if (statusClass) {
                element.classList.add(statusClass);
            }
        }
    }
    
    // Initial load
    fetchAgentProfiles(); // This will also trigger selectAgentForSession and then fetchSessionLLMParameters
    // fetchSessionLLMParameters(); // Called by fetchAgentProfiles after selection
    setEmptyState(plotOutlineContent, 'Plot details from the Book Plotter agent will appear here.', 'fas fa-feather-alt');
    setEmptyState(characterProfilesContent, 'Character details from the Character Developer agent will appear here.', 'fas fa-user-edit');
    setEmptyState(worldDetailsContent, 'World details from the World Builder agent will appear here.', 'fas fa-globe');
});