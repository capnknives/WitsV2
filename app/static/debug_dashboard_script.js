document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation
    const tabs = document.querySelectorAll('.tab-container .tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTabContentId = tab.dataset.tab + '-tab'; // e.g., 'llm-interface-perf-tab'
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Show selected tab content
            tabContents.forEach(content => {
                if (content.id === targetTabContentId) {
                    content.style.display = 'block'; // Show
                    content.classList.add('active');
                } else {
                    content.style.display = 'none'; // Hide
                    content.classList.remove('active');
                }
            });

            // Load data for the activated tab
            if (tab.dataset.tab.endsWith('-perf')) {
                fetchPerformanceData(tab.dataset.tab.replace('-perf', ''));
            } else if (tab.dataset.tab.endsWith('-logs')) {
                fetchLogData(tab.dataset.tab.replace('-logs', ''));
            }
        });
    });

    // Fetch and display system overview metrics
    function fetchSystemOverview() {
        fetch('/api/debug/metrics')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                const metricsContainer = document.getElementById('system-overview-metrics');
                metricsContainer.innerHTML = ''; // Clear previous metrics

                const metricsToShow = {
                    'active_sessions': 'Active Sessions',
                    'llm_calls': 'LLM Calls',
                    'tool_calls': 'Tool Calls',
                    'avg_response_time': 'Avg Response Time (ms)',
                    'memory_segments': 'Memory Segments'
                };

                for (const key in metricsToShow) {
                    if (data.hasOwnProperty(key)) {
                        const value = (key === 'avg_response_time') ? parseFloat(data[key]).toFixed(2) : data[key];
                        const metricItem = `
                            <div class="metric-item">
                                <div class="metric-value">${value}</div>
                                <div class="metric-label">${metricsToShow[key]}</div>
                            </div>`;
                        metricsContainer.innerHTML += metricItem;
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching system overview metrics:', error);
                const metricsContainer = document.getElementById('system-overview-metrics');
                metricsContainer.innerHTML = '<p>Error loading system overview data.</p>';
            });
    }

    // Fetch and display performance data for different components
    function fetchPerformanceData(component) {
        // Example: component can be 'llm_interface', 'memory_manager', 'tools', 'agents'
        fetch(`/api/debug/performance/${component}`)
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                const tableBodyId = `${component.replace('_', '-')}-table-body`; // e.g., llm-interface-table-body
                const tableBody = document.getElementById(tableBodyId);
                if (!tableBody) {
                    console.warn(`Table body with ID ${tableBodyId} not found.`);
                    return;
                }
                tableBody.innerHTML = ''; // Clear previous data

                if (data && data.length > 0) {
                    data.forEach(item => {
                        const row = tableBody.insertRow();
                        // This needs to be dynamic based on the component
                        if (component === 'llm_interface') {
                            row.insertCell().textContent = item.timestamp || 'N/A';
                            row.insertCell().textContent = item.action || 'N/A';
                            row.insertCell().textContent = item.model || 'N/A';
                            row.insertCell().textContent = item.duration_ms ? item.duration_ms.toFixed(2) : 'N/A';
                            row.insertCell().textContent = item.tokens ? `${item.tokens.prompt}/${item.tokens.completion}` : 'N/A';
                        } else if (component === 'memory_manager') {
                             row.insertCell().textContent = item.timestamp || 'N/A';
                             row.insertCell().textContent = item.action || 'N/A';
                             row.insertCell().textContent = item.duration_ms ? item.duration_ms.toFixed(2) : 'N/A';
                             row.insertCell().textContent = item.details ? JSON.stringify(item.details) : 'N/A';
                        } else if (component === 'tools') {
                            row.insertCell().textContent = item.timestamp || 'N/A';
                            row.insertCell().textContent = item.tool_name || 'N/A';
                            row.insertCell().textContent = item.action || 'N/A';
                            row.insertCell().textContent = item.duration_ms ? item.duration_ms.toFixed(2) : 'N/A';
                            row.insertCell().textContent = item.status || 'N/A';
                        } else if (component === 'agents') {
                            row.insertCell().textContent = item.timestamp || 'N/A';
                            row.insertCell().textContent = item.agent_name || 'N/A';
                            row.insertCell().textContent = item.action || 'N/A';
                            row.insertCell().textContent = item.duration_ms ? item.duration_ms.toFixed(2) : 'N/A';
                            row.insertCell().textContent = item.status || 'N/A';
                        }
                        // Add more specific data handling for other components if needed
                    });
                } else {
                    const row = tableBody.insertRow();
                    const cell = row.insertCell();
                    cell.colSpan = tableBody.previousElementSibling.rows[0].cells.length; // Colspan based on header
                    cell.textContent = 'No performance data available.';
                    cell.style.textAlign = 'center';
                }
            })
            .catch(error => {
                console.error(`Error fetching performance data for ${component}:`, error);
                const tableBody = document.getElementById(`${component.replace('_', '-')}-table-body`);
                if (tableBody) {
                    tableBody.innerHTML = `<tr><td colspan="5">Error loading ${component.replace('_', ' ')} performance data.</td></tr>`;
                }
            });
    }

    // Fetch and display log data
    function fetchLogData(logType) {
        // logType can be 'all', 'error', 'warning'
        fetch(`/api/debug/logs/${logType}`) // Assuming endpoint like /api/debug/logs/all
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                const logContentId = `${logType}-logs-content`;
                const preElement = document.getElementById(logContentId);
                if (!preElement) {
                    console.warn(`Log content element with ID ${logContentId} not found.`);
                    return;
                }
                if (data && data.logs && data.logs.length > 0) {
                    preElement.textContent = data.logs.map(log => 
                        `${log.timestamp} [${log.level}] ${log.component || ''} - ${log.message}`
                    ).join('\n');
                } else {
                    preElement.textContent = 'No logs available.';
                }
            })
            .catch(error => {
                console.error(`Error fetching ${logType} logs:`, error);
                const preElement = document.getElementById(`${logType}-logs-content`);
                if (preElement) {
                    preElement.textContent = `Error loading ${logType} logs.`;
                }
            });
    }

    // Initial data load
    fetchSystemOverview();
    // Load data for the initially active performance tab (e.g., LLM Interface)
    const activePerformanceTab = document.querySelector('#performance-tabs .tab.active');
    if (activePerformanceTab) {
        fetchPerformanceData(activePerformanceTab.dataset.tab.replace('-perf', ''));
    }
    // Load data for the initially active log tab (e.g., All Logs)
    const activeLogTab = document.querySelector('#log-tabs .tab.active');
    if (activeLogTab) {
        fetchLogData(activeLogTab.dataset.tab.replace('-logs', ''));
    }

    // Optional: Set up auto-refresh for some data
    // setInterval(fetchSystemOverview, 15000); // Refresh overview every 15 seconds
    // setInterval(() => { // Refresh active tabs
    //     const activePerf = document.querySelector('#performance-tabs .tab.active');
    //     if (activePerf) fetchPerformanceData(activePerf.dataset.tab.replace('-perf', ''));
    //     const activeLog = document.querySelector('#log-tabs .tab.active');
    //     if (activeLog) fetchLogData(activeLog.dataset.tab.replace('-logs', ''));
    // }, 30000); // Refresh logs/perf every 30 seconds
});

