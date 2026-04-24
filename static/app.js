function formatNumber(num) {
    return num ? num.toLocaleString() : '0';
}

function formatTime() {
    return new Date().toLocaleTimeString();
}

function formatDateTime(isoString) {
    if (!isoString) return '-';
    return new Date(isoString).toLocaleString();
}

function todayStr() {
    return new Date().toLocaleDateString('en-CA');
}

function daysAgoStr(n) {
    const d = new Date();
    d.setDate(d.getDate() - n);
    return d.toLocaleDateString('en-CA');
}

// Global filter state
let filterFrom = todayStr();
let filterTo = todayStr();

// Pagination state
let currentPage = 1;
const perPage = 20;
let totalPages = 1;

async function fetchStats(from, to) {
    try {
        let url = '/api/stats?';
        if (from) url += `from_date=${from}&`;
        if (to) url += `to_date=${to}`;
        const resp = await fetch(url);
        return await resp.json();
    } catch (e) {
        console.error('Failed to fetch stats:', e);
        return null;
    }
}

async function fetchHistory(from, to, page = 1) {
    try {
        const offset = (page - 1) * perPage;
        let url = `/api/history?limit=${perPage}&offset=${offset}`;
        if (from) url += `&from_date=${from}`;
        if (to) url += `&to_date=${to}`;
        const resp = await fetch(url);
        return await resp.json();
    } catch (e) {
        console.error('Failed to fetch history:', e);
        return null;
    }
}

function renderStats(data) {
    if (!data) return;

    const t = data.totals;
    document.getElementById('total-input').textContent = formatNumber(t.input);
    document.getElementById('total-output').textContent = formatNumber(t.output);
    document.getElementById('total-cache').textContent = formatNumber(t.cache);
    document.getElementById('total-all').textContent = formatNumber(t.total);
    document.getElementById('total-success').textContent = formatNumber(t.success_count);
    document.getElementById('total-fail').textContent = formatNumber(t.fail_count);
    document.getElementById('avg-duration').textContent = t.avg_duration_ms ? `${t.avg_duration_ms}ms` : '-';

    const tbody = document.getElementById('model-tbody');
    const models = data.models;

    if (Object.keys(models).length === 0) {
        tbody.innerHTML = '<tr><td colspan="9">No data</td></tr>';
        return;
    }

    let html = '';
    for (const [model, s] of Object.entries(models)) {
        html += `<tr>
            <td>${model}</td>
            <td>${formatNumber(s.input)}</td>
            <td>${formatNumber(s.output)}</td>
            <td>${formatNumber(s.cache)}</td>
            <td>${formatNumber(s.total)}</td>
            <td>${s.pct}</td>
            <td>${formatNumber(s.success_count)}</td>
            <td>${formatNumber(s.fail_count)}</td>
            <td>${s.avg_duration_ms ? s.avg_duration_ms + 'ms' : '-'}</td>
        </tr>`;
    }
    tbody.innerHTML = html;
}

let chartTokens = null;
let chartModelTokens = null;
let chartModelRequests = null;

const CHART_COLORS = ['#4fc3f7', '#ff8a65', '#81c784', '#ba68c8', '#ffd54f', '#f06292', '#4dd0e1', '#aed581'];

function makeChartOpts(textColor) {
    return { responsive: true, plugins: { legend: { labels: { color: textColor } } } };
}

function renderCharts(data) {
    if (!data) return;
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#e0e0e0' : '#333';

    const t = data.totals;
    const tokenData = [t.input, t.output, t.cache];

    // Token distribution
    if (chartTokens) {
        chartTokens.data.datasets[0].data = tokenData;
        chartTokens.options.plugins.legend.labels.color = textColor;
        chartTokens.update('none');
    } else {
        chartTokens = new Chart(document.getElementById('chart-tokens'), {
            type: 'doughnut',
            data: {
                labels: ['Input', 'Output', 'Cache'],
                datasets: [{ data: tokenData, backgroundColor: ['#4fc3f7', '#ff8a65', '#81c784'], borderWidth: 0 }]
            },
            options: makeChartOpts(textColor)
        });
    }

    // Per model
    const models = Object.entries(data.models);
    const modelLabels = models.map(([m]) => m);
    const modelTokenData = models.map(([, s]) => s.total);
    const modelRequestData = models.map(([, s]) => s.count);
    const colors = CHART_COLORS.slice(0, modelLabels.length);

    if (chartModelTokens) {
        chartModelTokens.data.labels = modelLabels;
        chartModelTokens.data.datasets[0].data = modelTokenData;
        chartModelTokens.data.datasets[0].backgroundColor = colors;
        chartModelTokens.options.plugins.legend.labels.color = textColor;
        chartModelTokens.update('none');
    } else {
        chartModelTokens = new Chart(document.getElementById('chart-model-tokens'), {
            type: 'doughnut',
            data: {
                labels: modelLabels,
                datasets: [{ data: modelTokenData, backgroundColor: colors, borderWidth: 0 }]
            },
            options: makeChartOpts(textColor)
        });
    }

    if (chartModelRequests) {
        chartModelRequests.data.labels = modelLabels;
        chartModelRequests.data.datasets[0].data = modelRequestData;
        chartModelRequests.data.datasets[0].backgroundColor = colors;
        chartModelRequests.options.plugins.legend.labels.color = textColor;
        chartModelRequests.update('none');
    } else {
        chartModelRequests = new Chart(document.getElementById('chart-model-requests'), {
            type: 'doughnut',
            data: {
                labels: modelLabels,
                datasets: [{ data: modelRequestData, backgroundColor: colors, borderWidth: 0 }]
            },
            options: makeChartOpts(textColor)
        });
    }
}

function renderHistory(data) {
    const tbody = document.getElementById('history-tbody');
    if (!data || data.logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9">No history</td></tr>';
        updatePagination(1, 1);
        return;
    }

    let html = '';
    for (const log of data.logs) {
        const duration = log.duration_ms ? `${log.duration_ms}ms` : '-';
        const status = log.success ? '<span class="status-ok">&#10004;</span>' : '<span class="status-fail">&#10008;</span>';
        const error = log.error ? `<span class="error-text">${log.error}</span>` : '-';
        html += `<tr>
            <td>${formatDateTime(log.timestamp)}</td>
            <td>${log.original_model || '-'}</td>
            <td>${log.model || '-'}</td>
            <td>${duration}</td>
            <td>${formatNumber(log.tokens_input)}</td>
            <td>${formatNumber(log.tokens_output)}</td>
            <td>${formatNumber(log.tokens_cache)}</td>
            <td>${status}</td>
            <td>${error}</td>
        </tr>`;
    }
    tbody.innerHTML = html;

    // Update pagination
    totalPages = Math.ceil(data.total / perPage);
    updatePagination(currentPage, totalPages);
}

function updatePagination(current, total) {
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');

    pageInfo.textContent = `Page ${current} of ${total}`;
    prevBtn.disabled = current <= 1;
    nextBtn.disabled = current >= total;
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });
}

function setupFilter() {
    const rangeBtns = document.querySelectorAll('.filter-bar .btn[data-range]');
    const customFields = [document.getElementById('from-date'), document.getElementById('to-date'),
                          document.getElementById('date-sep'), document.getElementById('apply-filter')];

    function setActiveRange(range) {
        rangeBtns.forEach(b => b.classList.toggle('active', b.dataset.range === range));
        const isCustom = range === 'custom';
        customFields.forEach(el => el.style.display = isCustom ? '' : 'none');
    }

    rangeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const range = btn.dataset.range;
            setActiveRange(range);
            currentPage = 1; // Reset to first page when filter changes
            if (range === 'today') { filterFrom = todayStr(); filterTo = todayStr(); }
            else if (range === '7d') { filterFrom = daysAgoStr(7); filterTo = todayStr(); }
            else if (range === '30d') { filterFrom = daysAgoStr(30); filterTo = todayStr(); }
            else if (range === 'custom') {
                document.getElementById('from-date').value = filterFrom;
                document.getElementById('to-date').value = filterTo;
                return;
            }
            refreshAll();
        });
    });

    document.getElementById('apply-filter').addEventListener('click', () => {
        filterFrom = document.getElementById('from-date').value;
        filterTo = document.getElementById('to-date').value;
        currentPage = 1;
        refreshAll();
    });
}

async function refreshAll() {
    const [stats, history] = await Promise.all([
        fetchStats(filterFrom, filterTo),
        fetchHistory(filterFrom, filterTo, currentPage)
    ]);
    renderStats(stats);
    renderCharts(stats);
    renderHistory(history);
    document.getElementById('last-update').textContent = `Last update: ${formatTime()}`;
}

async function loadHistory() {
    const history = await fetchHistory(filterFrom, filterTo, currentPage);
    renderHistory(history);
}

async function deleteHistory(before = null, all = false) {
    try {
        let url = '/api/history?';
        if (all) url += 'all=true';
        else {
            const d = before || filterTo;
            url += `before=${d}`;
        }
        await fetch(url, { method: 'DELETE' });
        currentPage = 1;
        refreshAll();
    } catch (e) {
        console.error('Failed to delete:', e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Theme
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    const toggle = document.getElementById('theme-toggle');
    toggle.textContent = saved === 'dark' ? '☾' : '☀';
    toggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        toggle.textContent = next === 'dark' ? '☾' : '☀';
    });

    setupTabs();
    setupFilter();
    refreshAll();
    let _refreshTimer = setInterval(refreshAll, 15000);
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(_refreshTimer);
        } else {
            refreshAll();
            _refreshTimer = setInterval(refreshAll, 15000);
        }
    });

    const deleteBtn = document.getElementById('delete-btn');
    const deleteMenu = document.getElementById('delete-menu');
    const deleteDate = document.getElementById('delete-date');

    deleteBtn.addEventListener('click', () => {
        deleteMenu.style.display = deleteMenu.style.display === 'none' ? '' : 'none';
    });

    document.getElementById('delete-all-opt').addEventListener('click', () => {
        if (confirm('Delete all history?')) {
            deleteHistory(null, true);
        }
        deleteMenu.style.display = 'none';
    });

    document.getElementById('delete-by-date-opt').addEventListener('click', () => {
        deleteMenu.style.display = 'none';
        document.getElementById('delete-date').value = todayStr();
        document.getElementById('delete-modal').style.display = '';
    });

    document.getElementById('modal-delete-btn').addEventListener('click', () => {
        const d = document.getElementById('delete-date').value;
        if (d && confirm(`Delete history before ${d}?`)) {
            deleteHistory(d);
        }
        document.getElementById('delete-modal').style.display = 'none';
    });

    document.getElementById('modal-cancel-btn').addEventListener('click', () => {
        document.getElementById('delete-modal').style.display = 'none';
    });

    // Pagination buttons
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadHistory();
        }
    });

    document.getElementById('next-page').addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadHistory();
        }
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.delete-section')) {
            deleteMenu.style.display = 'none';
        }
    });
});