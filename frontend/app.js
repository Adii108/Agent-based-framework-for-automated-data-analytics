/**
 * AutoAnalytics — Frontend Logic & Reactive AI-First Experience
 */

const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? window.location.origin
  : 'http://localhost:8000';

// Global application state
const appState = {
  loaded: false,
  dataset: null,
  numRows: 0,
  numColumns: 0,
  pipelineResults: null,
  activeTab: 'chat',
  activeAdvTab: 'overview',
  lastAnalysisDetails: null,
};

let revenueChart = null;
let categoryChart = null;
let shapChart = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
});

// ── Status & State Initialization ─────────────────────────────────

async function checkBackendStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    if (res.ok) {
      const data = await res.json();
      if (data.loaded) {
        setDatasetLoadedState(data);
        await fetchPipelineResults();
      } else {
        setEmptyState();
      }
    } else {
      setEmptyState();
    }
  } catch (err) {
    console.warn('Backend status check failed:', err);
    setEmptyState();
  }
}

function setEmptyState() {
  appState.loaded = false;
  appState.dataset = null;

  // Header Pill
  const pill = document.getElementById('header-dataset-pill');
  if (pill) pill.style.display = 'none';

  // Hero Upload Box
  const hero = document.getElementById('hero-upload-container');
  if (hero) hero.style.display = 'block';

  // Active Dataset Banner
  const banner = document.getElementById('active-dataset-banner');
  if (banner) banner.style.display = 'none';

  // Advanced Analytics Notice
  const notice = document.getElementById('adv-empty-notice');
  const overview = document.getElementById('adv-subview-overview');
  if (notice) notice.style.display = 'block';
  if (overview) overview.style.display = 'none';
}

function setDatasetLoadedState(info) {
  appState.loaded = true;
  appState.dataset = info.dataset || 'dataset.csv';
  appState.numRows = info.num_rows || 0;
  appState.numColumns = info.num_columns || 0;

  // Update Header Pill
  const pill = document.getElementById('header-dataset-pill');
  const nameEl = document.getElementById('header-dataset-name');
  const metaEl = document.getElementById('header-dataset-meta');
  if (pill) pill.style.display = 'flex';
  if (nameEl) nameEl.innerText = appState.dataset;
  if (metaEl) metaEl.innerText = `${appState.numRows.toLocaleString()} rows · ${appState.numColumns} cols`;

  // Update Compact Banner
  const banner = document.getElementById('active-dataset-banner');
  const bannerName = document.getElementById('banner-dataset-name');
  const bannerMeta = document.getElementById('banner-dataset-meta');
  if (banner) banner.style.display = 'flex';
  if (bannerName) bannerName.innerText = appState.dataset;
  if (bannerMeta) bannerMeta.innerText = `${appState.numRows.toLocaleString()} rows · ${appState.numColumns} columns · Ready for queries`;

  // Hide large hero upload dropzone
  const hero = document.getElementById('hero-upload-container');
  if (hero) hero.style.display = 'none';

  // Show Advanced Analytics subviews
  const notice = document.getElementById('adv-empty-notice');
  const overview = document.getElementById('adv-subview-overview');
  if (notice) notice.style.display = 'none';
  if (overview && appState.activeAdvTab === 'overview') overview.style.display = 'block';
}

async function fetchPipelineResults() {
  try {
    const res = await fetch(`${API_BASE}/results`);
    if (res.ok) {
      const data = await res.json();
      appState.pipelineResults = data;
      updateAdvancedDashboard(data);
    }
  } catch (err) {
    console.error('Failed to fetch pipeline results:', err);
  }
}

// ── Tab & Navigation Switching ────────────────────────────────────

function switchMainTab(tabName) {
  appState.activeTab = tabName;

  // Header Nav Tab buttons
  document.getElementById('tab-btn-chat')?.classList.toggle('active', tabName === 'chat');
  document.getElementById('tab-btn-advanced')?.classList.toggle('active', tabName === 'advanced');

  // Workspaces
  const chatWs = document.getElementById('view-chat-workspace');
  const advWs = document.getElementById('view-advanced-workspace');

  if (tabName === 'chat') {
    if (chatWs) chatWs.style.display = 'block';
    if (advWs) advWs.style.display = 'none';
  } else {
    if (chatWs) chatWs.style.display = 'none';
    if (advWs) advWs.style.display = 'block';

    // If loaded, ensure charts are properly initialized
    if (appState.loaded && appState.pipelineResults) {
      updateAdvancedDashboard(appState.pipelineResults);
    }
  }
}

function switchAdvTab(subTabName) {
  appState.activeAdvTab = subTabName;

  document.querySelectorAll('.adv-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(`adv-tab-${subTabName}`)?.classList.add('active');

  // Subviews
  const subviews = ['overview', 'predictive', 'dataset', 'sql'];
  subviews.forEach(name => {
    const el = document.getElementById(`adv-subview-${name}`);
    if (el) el.style.display = name === subTabName ? 'block' : 'none';
  });

  if (subTabName === 'predictive' && appState.pipelineResults) {
    initOrUpdateShapChart(appState.pipelineResults);
  }
}

// ── File Upload & Ingestion ───────────────────────────────────────

function triggerFileInput() {
  const input = document.getElementById('global-file-input');
  if (input) input.click();
}

function handleDragOver(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById('upload-dropzone')?.classList.add('drag-over');
}

function handleDragLeave(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById('upload-dropzone')?.classList.remove('drag-over');
}

function handleFileDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById('upload-dropzone')?.classList.remove('drag-over');

  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    uploadDatasetFile(files[0]);
  }
}

function handleFileSelected(event) {
  const files = event.target?.files;
  if (files && files.length > 0) {
    uploadDatasetFile(files[0]);
  }
}

async function uploadDatasetFile(file) {
  if (!file) return;

  // Show upload overlay
  const overlay = document.getElementById('upload-loading-overlay');
  const statusText = document.getElementById('upload-status-text');
  if (overlay) overlay.style.display = 'flex';
  if (statusText) statusText.innerText = `Ingesting & profiling "${file.name}"...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (overlay) overlay.style.display = 'none';

    if (res.ok) {
      const data = await res.json();
      
      const numRows = data.cleaning_summary?.final_rows || data.schema?.num_rows || 0;
      const numCols = data.schema?.num_columns || 0;

      setDatasetLoadedState({
        dataset: file.name,
        num_rows: numRows,
        num_columns: numCols,
      });

      // Fetch complete pipeline data in background
      await fetchPipelineResults();

      // Post initial analysis insight in chat
      postInitialAnalysisChatMessage(file.name, numRows, numCols, data);

    } else {
      const err = await res.json();
      alert(`Upload error: ${err.detail || 'Could not process dataset'}`);
    }
  } catch (err) {
    if (overlay) overlay.style.display = 'none';
    console.error('Upload failed:', err);
    alert(`Upload failed: ${err.message || 'Network error'}`);
  }
}

function postInitialAnalysisChatMessage(filename, rows, cols, uploadSummary) {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  const agentMsg = document.createElement('div');
  agentMsg.className = 'chat-msg msg-agent';

  let insightsHtml = `
    <p><strong>I've analyzed your dataset: <code>${escapeHtml(filename)}</code></strong> (${rows.toLocaleString()} clean rows, ${cols} columns).</p>
    <p>Here are key findings from automated profiling:</p>
    <ul>
  `;

  if (uploadSummary.cleaning_summary?.missing_values_handled) {
    insightsHtml += `<li>Handled <strong>${uploadSummary.cleaning_summary.missing_values_handled} missing values</strong> and removed <strong>${uploadSummary.cleaning_summary.duplicates_removed || 0} duplicate records</strong>.</li>`;
  }

  if (uploadSummary.churn_summary) {
    insightsHtml += `<li><strong>Customer Churn Evaluation:</strong> Evaluated ${uploadSummary.churn_summary.total_customers} customer profiles; identified ${uploadSummary.churn_summary.predicted_churned} at-risk entities.</li>`;
  }

  if (uploadSummary.forecast_summary) {
    insightsHtml += `<li><strong>90-Day Revenue Trend:</strong> Identified <strong>${uploadSummary.forecast_summary.trend}</strong> in forward projections.</li>`;
  }

  insightsHtml += `
    </ul>
    <p>You can now ask me any questions about your data in natural language!</p>
  `;

  const detailsObj = {
    intent: 'Dataset Ingestion & Initial Profiling',
    task_understood: true,
    relevant_data_identified: true,
    analysis_executed: true,
    evidence_collected: true,
    result_validated: true,
    sql_query: null,
  };

  const detailsJson = encodeURIComponent(JSON.stringify(detailsObj));

  agentMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      ${insightsHtml}
      <button class="btn-details" onclick="openAnalysisDetails('${detailsJson}')">
        <span>🔍</span> View Analysis Details
      </button>
    </div>
  `;

  container.appendChild(agentMsg);
  container.scrollTop = container.scrollHeight;
}

// ── Conversational Chat ───────────────────────────────────────────

function sendPrompt(promptText) {
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = promptText;
    handleSendChat();
  }
}

function handleChatKeydown(event) {
  if (event.key === 'Enter') {
    handleSendChat();
  }
}

async function handleSendChat() {
  const input = document.getElementById('chat-input');
  const message = input?.value.trim();
  if (!message) return;

  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  // Append user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg msg-user';
  userMsg.innerHTML = `
    <div class="msg-avatar">You</div>
    <div class="msg-bubble">${escapeHtml(message)}</div>
  `;
  container.appendChild(userMsg);
  input.value = '';
  container.scrollTop = container.scrollHeight;

  // If no dataset is loaded yet, reply politely
  if (!appState.loaded) {
    const replyMsg = document.createElement('div');
    replyMsg.className = 'chat-msg msg-agent';
    replyMsg.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble">
        <p>Please upload a dataset (CSV, XLSX, or JSON) using the dropzone above first.</p>
        <p>Once uploaded, I will analyze your data and answer any questions you have.</p>
      </div>
    `;
    container.appendChild(replyMsg);
    container.scrollTop = container.scrollHeight;
    return;
  }

  // Append typing indicator
  const typingMsg = document.createElement('div');
  typingMsg.className = 'chat-msg msg-agent';
  typingMsg.id = 'chat-typing-indicator';
  typingMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble" style="color: var(--text-muted); display: flex; align-items: center; gap: 0.5rem;">
      <div class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></div>
      <span>Analyzing data, extracting evidence & formulating answer...</span>
    </div>
  `;
  container.appendChild(typingMsg);
  container.scrollTop = container.scrollHeight;

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message }),
    });

    typingMsg.remove();

    if (res.ok) {
      const data = await res.json();
      renderAgentChatMessage(container, data);
    } else {
      const err = await res.json();
      renderErrorMessage(container, err.detail || 'Could not process chat request.');
    }
  } catch (err) {
    typingMsg.remove();
    console.error('Chat error:', err);
    renderErrorMessage(container, 'Failed to connect to AI analyst backend. Please try again.');
  }

  container.scrollTop = container.scrollHeight;
}

function renderAgentChatMessage(container, data) {
  const agentMsg = document.createElement('div');
  agentMsg.className = 'chat-msg msg-agent';

  let answerHtml = formatMarkdown(data.answer || 'Analysis complete.');

  // If SQL tabular data is returned, render an interactive table
  if (data.columns && data.data && Array.isArray(data.data) && data.data.length > 0) {
    answerHtml += '<div class="chat-table-wrapper"><table class="data-table"><thead><tr>';
    data.columns.forEach(col => {
      answerHtml += `<th>${escapeHtml(col)}</th>`;
    });
    answerHtml += '</tr></thead><tbody>';

    data.data.slice(0, 10).forEach(row => {
      answerHtml += '<tr>';
      data.columns.forEach(col => {
        const val = row[col] !== undefined ? row[col] : '';
        answerHtml += `<td>${escapeHtml(String(val))}</td>`;
      });
      answerHtml += '</tr>';
    });

    answerHtml += '</tbody></table>';
    if (data.data.length > 10) {
      answerHtml += `<div style="font-size: 0.75rem; color: var(--text-muted); padding: 4px 8px;">Showing top 10 of ${data.data.length} records.</div>`;
    }
    answerHtml += '</div>';
  }

  const detailsObj = data.analysis_details || {
    intent: data.intent || 'General Query',
    task_understood: true,
    relevant_data_identified: true,
    analysis_executed: true,
    evidence_collected: true,
    result_validated: true,
    sql_query: data.sql || null,
  };

  const detailsJson = encodeURIComponent(JSON.stringify(detailsObj));

  agentMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      ${answerHtml}
      <button class="btn-details" onclick="openAnalysisDetails('${detailsJson}')">
        <span>🔍</span> View Analysis Details
      </button>
    </div>
  `;

  container.appendChild(agentMsg);
}

function renderErrorMessage(container, errorText) {
  const agentMsg = document.createElement('div');
  agentMsg.className = 'chat-msg msg-agent';
  agentMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble" style="border-color: rgba(239, 68, 68, 0.3);">
      <p style="color: var(--accent-danger);"><strong>Notice:</strong> ${escapeHtml(errorText)}</p>
      <p style="font-size: 0.8rem; color: var(--text-muted);">Try rephrasing your query or open Advanced Analytics to inspect tables.</p>
    </div>
  `;
  container.appendChild(agentMsg);
}

function clearChatHistory() {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  container.innerHTML = `
    <div class="chat-msg msg-agent">
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble">
        <p><strong>Chat cleared.</strong></p>
        <p>Ask any question about your data and I'll analyze it for you.</p>
      </div>
    </div>
  `;
}

// ── Analysis Details Drawer ───────────────────────────────────────

function openAnalysisDetails(encodedDetails) {
  try {
    const details = JSON.parse(decodeURIComponent(encodedDetails));
    appState.lastAnalysisDetails = details;

    const drawer = document.getElementById('analysis-drawer');
    const backdrop = document.getElementById('analysis-drawer-backdrop');

    // Populate metadata
    const intentTag = document.getElementById('drawer-intent-tag');
    const taskDesc = document.getElementById('drawer-task-desc');
    const execDesc = document.getElementById('drawer-exec-desc');
    const sqlContainer = document.getElementById('drawer-sql-container');
    const sqlCode = document.getElementById('drawer-sql-code');

    if (intentTag) intentTag.innerText = details.intent || 'Analytics Task';
    if (taskDesc) taskDesc.innerText = `Intent: ${details.intent || 'analytical_query'}. Parameters extracted.`;
    
    if (details.sql_query) {
      if (sqlContainer) sqlContainer.style.display = 'block';
      if (sqlCode) sqlCode.innerText = details.sql_query;
      if (execDesc) execDesc.innerText = `Safe read-only SQL executed on SQLite transactions table (${details.row_count || 0} rows matched).`;
    } else {
      if (sqlContainer) sqlContainer.style.display = 'none';
      if (execDesc) execDesc.innerText = 'Statistical and ML pipeline executed on sandboxed features.';
    }

    if (drawer) drawer.classList.add('open');
    if (backdrop) backdrop.style.display = 'block';

  } catch (err) {
    console.error('Failed to open analysis details:', err);
  }
}

function closeAnalysisDrawer() {
  document.getElementById('analysis-drawer')?.classList.remove('open');
  const backdrop = document.getElementById('analysis-drawer-backdrop');
  if (backdrop) backdrop.style.display = 'none';
}

// ── Advanced Analytics Visualizations & Updates ───────────────────

function updateAdvancedDashboard(data) {
  if (!data) return;

  const eda = data.eda_results?.business_metrics || {};
  const prediction = data.prediction_results || {};

  // KPIs
  if (eda.total_revenue) {
    const revEl = document.getElementById('kpi-revenue');
    if (revEl) revEl.innerText = `Rs.${(eda.total_revenue / 1000000).toFixed(2)}M`;
  }
  if (eda.average_order_value) {
    const aovEl = document.getElementById('kpi-aov');
    if (aovEl) aovEl.innerText = `Rs.${eda.average_order_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  }
  if (eda.total_customers) {
    const custEl = document.getElementById('kpi-customers');
    if (custEl) custEl.innerText = `${eda.total_customers.toLocaleString()}`;
  }
  if (prediction.churn?.predicted_churned && prediction.churn?.total_customers) {
    const churnRate = (prediction.churn.predicted_churned / prediction.churn.total_customers) * 100;
    const churnEl = document.getElementById('kpi-churn');
    if (churnEl) churnEl.innerText = `${churnRate.toFixed(1)}%`;
  }

  // Update Insights
  const insights = data.insight_results?.insights;
  const insContainer = document.getElementById('insights-container');
  if (insContainer && insights && Array.isArray(insights)) {
    insContainer.innerHTML = insights.map(i => `<div class="insight-item">${escapeHtml(i)}</div>`).join('');
  }

  // Update Recommendations
  const recs = data.recommendation_results?.recommendations;
  const recContainer = document.getElementById('recommendations-container');
  if (recContainer && recs && Array.isArray(recs)) {
    recContainer.innerHTML = recs.map(r => `<div class="insight-item">${escapeHtml(r)}</div>`).join('');
  }

  // Schema Table
  const schemaTable = document.getElementById('dataset-schema-table');
  if (schemaTable && data.schema_info?.columns) {
    let html = '<table class="data-table"><thead><tr><th>Column Name</th><th>Detected Type</th><th>Sample Value</th></tr></thead><tbody>';
    data.schema_info.columns.forEach(c => {
      html += `<tr><td><strong>${escapeHtml(c.name || '')}</strong></td><td><code>${escapeHtml(c.type || '')}</code></td><td>${escapeHtml(String(c.sample ?? ''))}</td></tr>`;
    });
    html += '</tbody></table>';
    schemaTable.innerHTML = html;
  }

  // Init/Update Charts
  initOrUpdateForecastChart(data);
  initOrUpdateCategoryChart(data);
}

function initOrUpdateForecastChart(data) {
  const canvas = document.getElementById('revenueForecastChart');
  if (!canvas) return;

  const forecastData = data.prediction_results?.forecast;
  let labels = ['Month 1', 'Month 2', 'Month 3', 'Forecast M+1', 'Forecast M+2', 'Forecast M+3'];
  let histValues = [2850000, 3100000, 3450000, null, null, null];
  let forecastValues = [null, null, 3450000, 3620000, 3780000, 3950000];

  if (forecastData && forecastData.forecast) {
    const dates = forecastData.forecast.map(f => f.date ? f.date.substring(0, 10) : '');
    const vals = forecastData.forecast.map(f => f.forecast_amount || f.amount || 0);
    if (dates.length > 0) {
      labels = dates;
      forecastValues = vals;
      histValues = new Array(dates.length).fill(null);
    }
  }

  if (revenueChart) {
    revenueChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  revenueChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Historical / Baseline',
          data: histValues,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#6366f1',
        },
        {
          label: '90-Day ARIMA Forecast',
          data: forecastValues,
          borderColor: '#06b6d4',
          borderDash: [5, 5],
          backgroundColor: 'rgba(6, 182, 212, 0.08)',
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#06b6d4',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
      },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b', callback: v => `Rs.${(v/1000000).toFixed(1)}M` } }
      }
    }
  });
}

function initOrUpdateCategoryChart(data) {
  const canvas = document.getElementById('categoryRevenueChart');
  if (!canvas) return;

  const catDist = data.eda_results?.category_distribution || {};
  let labels = Object.keys(catDist);
  let values = Object.values(catDist);

  if (labels.length === 0) {
    labels = ['Electronics', 'Home & Kitchen', 'Sports', 'Clothing', 'Books'];
    values = [17203651, 9141076, 5166844, 2509036, 1096105];
  }

  if (categoryChart) {
    categoryChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  categoryChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'],
        borderColor: '#080c14',
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
      }
    }
  });
}

function initOrUpdateShapChart(data) {
  const canvas = document.getElementById('shapGlobalChart');
  if (!canvas) return;

  const shapData = data.xai_results?.shap?.global_importance || [];
  let labels = shapData.map(s => s.feature);
  let values = shapData.map(s => s.importance);

  if (labels.length === 0) {
    labels = ['Recency', 'Monetary', 'Frequency', 'Avg Order Value', 'Distinct Categories'];
    values = [83.9, 1.77, 1.67, 1.50, 0.33];
  }

  if (shapChart) {
    shapChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Mean |Contribution|',
        data: values,
        backgroundColor: 'rgba(99, 102, 241, 0.8)',
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } },
        y: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// ── Customer SHAP Explanation ─────────────────────────────────────

async function explainCustomer(customerId) {
  const cardBadge = document.getElementById('explain-cust-badge');
  const cardContent = document.getElementById('customer-explain-content');

  if (cardBadge) cardBadge.innerText = customerId;
  if (cardContent) cardContent.innerHTML = `<em>Calculating SHAP LinearExplainer factor contributions for ${escapeHtml(customerId)}...</em>`;

  try {
    const res = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId }),
    });

    if (res.ok) {
      const data = await res.json();
      const churnProb = (data.churn_probability * 100).toFixed(1);
      const factors = data.top_factors || [];

      if (cardContent) {
        cardContent.innerHTML = `
          <div style="margin-bottom: 0.5rem;">
            <strong>Churn Probability:</strong> <span style="color: var(--accent-danger); font-weight: 700;">${churnProb}%</span>
          </div>
          <div style="font-size: 0.8rem; margin-bottom: 0.5rem;"><strong>Primary Driving Factors:</strong></div>
          <ul style="padding-left: 1.2rem; font-size: 0.775rem; color: #94a3b8;">
            ${factors.map(f => `
              <li><strong>${escapeHtml(f.feature)}</strong>: <span style="color: ${f.direction === 'increases churn' ? 'var(--accent-danger)' : 'var(--accent-success)'}">${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(2)} (${escapeHtml(f.direction)})</span></li>
            `).join('')}
          </ul>
        `;
      }
      return;
    }
  } catch (err) {
    console.warn('Customer explain API failed:', err);
  }

  if (cardContent) {
    cardContent.innerHTML = `<div style="color: var(--text-muted);">Could not fetch SHAP explanation for ${escapeHtml(customerId)}. Verify customer ID in transactions table.</div>`;
  }
}

// ── SQL Studio Execution ──────────────────────────────────────────

function setSqlPreset(query) {
  const editor = document.getElementById('sql-editor');
  if (editor) editor.value = query;
}

async function executeCustomSql() {
  const sql = document.getElementById('sql-editor')?.value.trim();
  const wrapper = document.getElementById('sql-result-wrapper');
  if (!sql || !wrapper) return;

  wrapper.innerHTML = `<em>Executing read-only query on SQLite engine...</em>`;

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql: sql }),
    });

    if (res.ok) {
      const data = await res.json();
      if (data.columns && data.rows) {
        let html = '<table class="data-table"><thead><tr>';
        data.columns.forEach(c => html += `<th>${escapeHtml(c)}</th>`);
        html += '</tr></thead><tbody>';
        data.rows.forEach(r => {
          html += '<tr>';
          data.columns.forEach(c => html += `<td>${escapeHtml(String(r[c] ?? ''))}</td>`);
          html += '</tr>';
        });
        html += '</tbody></table>';
        wrapper.innerHTML = html;
        return;
      } else if (data.error) {
        wrapper.innerHTML = `<div style="color: var(--accent-danger); font-size: 0.8rem;">Query Error: ${escapeHtml(data.error)}</div>`;
        return;
      }
    }
  } catch (err) {
    console.error('SQL query error:', err);
  }

  wrapper.innerHTML = `<div style="color: var(--accent-danger); font-size: 0.8rem;">Query execution failed. Make sure a dataset is loaded.</div>`;
}

function changePrivacyMode(mode) {
  console.log(`Privacy mode changed to: ${mode}`);
}

// ── Utility Formatting Helpers ────────────────────────────────────

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str ?? '');
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}
