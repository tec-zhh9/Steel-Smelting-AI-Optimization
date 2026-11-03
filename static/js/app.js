let currentPage = 1;
let previewPages = 1;
let trainTimer = null;
let cleanChart = null, lossChart = null, optChart = null;

function gotoPage(name) {
  document.querySelectorAll('.menu-item').forEach(m =>
    m.classList.toggle('active', m.dataset.page === name));
  document.querySelectorAll('.page').forEach(p =>
    p.classList.toggle('active', p.id === 'page-' + name));
  if (name === 'data') refreshDatasets();
  if (name === 'train') refreshTrainForm();
  if (name === 'optimize') refreshOptimize();
}
document.querySelectorAll('.menu-item').forEach(m =>
  m.addEventListener('click', () => gotoPage(m.dataset.page)));

function toast(msg, ok = true) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.background = ok ? '#2F78FF' : '#f56c6c';
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.style.opacity = 1);
  setTimeout(() => { t.style.opacity = 0; setTimeout(() => t.remove(), 400); }, 2500);
}

async function loadStats() {
  const r = await fetch('/api/stats').then(x => x.json());
  document.getElementById('stat-datasets').textContent = r.datasets;
  document.getElementById('stat-samples').textContent = r.samples;
  document.getElementById('stat-models').textContent = r.models;
  document.getElementById('stat-opts').textContent = r.optimizations;
}

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) doUpload(fileInput.files[0]); });

async function doUpload(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { toast(j.detail || '上传失败', false); return; }
  document.getElementById('upload-info').innerHTML =
    `${file.name}（${(file.size / 1024).toFixed(1)} KB，${j.rows} 行 × ${j.cols} 列）`;
  toast('上传成功');
  await refreshDatasets(j.name);
  loadStats();
}

async function refreshDatasets(selectName) {
  const j = await fetch('/api/datasets').then(x => x.json());
  const sel = document.getElementById('dataset-select');
  sel.innerHTML = '';
  j.datasets.forEach(d => {
    const o = document.createElement('option');
    o.value = d.name;
    o.textContent = `${d.name}（${d.rows}行${d.cleaned ? '，已清洗' : ''}）`;
    sel.appendChild(o);
  });
  if (selectName) sel.value = selectName;
  if (j.datasets.length) loadPreview(1);
}

async function loadPreview(page) {
  const name = document.getElementById('dataset-select').value;
  if (!name) return;
  currentPage = page;
  const j = await fetch(`/api/preview?name=${encodeURIComponent(name)}&page=${page}`).then(x => x.json());
  previewPages = j.pages;
  document.getElementById('page-info').textContent = `${j.page} / ${j.pages}`;

  const table = document.getElementById('preview-table');
  let html = '<thead><tr>' + j.columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  j.rows.forEach(row => {
    html += '<tr>' + row.map(v => `<td>${typeof v === 'number' ? +v.toFixed(4) : v}</td>`).join('') + '</tr>';
  });
  table.innerHTML = html + '</tbody>';
}

function prevPage() { if (currentPage > 1) loadPreview(currentPage - 1); }
function nextPage() { if (currentPage < previewPages) loadPreview(currentPage + 1); }

async function runClean() {
  const name = document.getElementById('dataset-select').value;
  if (!name) { toast('请先选择数据集', false); return; }
  const fd = new FormData();
  fd.append('dataset', name);
  fd.append('missing', document.getElementById('clean-missing').value);
  fd.append('outliers', document.getElementById('clean-outliers').checked);
  fd.append('standardize', document.getElementById('clean-standardize').checked);
  const r = await fetch('/api/clean', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { toast(j.detail || '清洗失败', false); return; }

  document.getElementById('clean-stats').style.display = 'flex';
  document.getElementById('clean-missing-count').textContent = j.missing_before;
  document.getElementById('clean-outlier-count').textContent = j.outliers_removed;
  document.getElementById('clean-rows').textContent = j.rows_after;

  if (!cleanChart) cleanChart = echarts.init(document.getElementById('clean-chart'));
  cleanChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['清洗前均值', '清洗后均值'] },
    xAxis: { type: 'category', data: j.before.columns, axisLabel: { rotate: 40, fontSize: 10 } },
    yAxis: { type: 'value' },
    grid: { left: 60, right: 20, bottom: 80 },
    series: [
      { name: '清洗前均值', type: 'bar', data: j.before.mean, itemStyle: { color: '#C0C4CC' } },
      { name: '清洗后均值', type: 'bar', data: j.after.mean, itemStyle: { color: '#2F78FF' } }
    ]
  });
  toast('清洗完成');
  refreshDatasets(name);
}

const ALGOS = {
  ml: [['random_forest', '随机森林'], ['xgboost', 'XGBoost'], ['svr', '支持向量机']],
  dl: [['bp', 'BP 神经网络'], ['lstm', 'LSTM']]
};
let trainCat = 'ml';

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  trainCat = t.dataset.cat;
  fillAlgos();
}));

function fillAlgos() {
  const sel = document.getElementById('train-algo');
  sel.innerHTML = '';
  ALGOS[trainCat].forEach(([v, label]) => {
    const o = document.createElement('option');
    o.value = v; o.textContent = label;
    sel.appendChild(o);
  });
  syncLrDefault();
}

function syncLrDefault() {
  const algo = document.getElementById('train-algo').value;
  if (algo === 'lstm') document.getElementById('train-lr').value = 0.001;
}
document.getElementById('train-algo').addEventListener('change', syncLrDefault);

async function refreshTrainForm() {
  fillAlgos();
  const j = await fetch('/api/datasets').then(x => x.json());
  const sel = document.getElementById('train-dataset');
  sel.innerHTML = '';
  j.datasets.forEach(d => {
    const o = document.createElement('option');
    o.value = d.name;
    o.textContent = d.name + (d.cleaned ? '（已清洗）' : '');
    sel.appendChild(o);
  });
}

async function startTrain() {
  const fd = new FormData();
  fd.append('dataset', document.getElementById('train-dataset').value);
  fd.append('category', trainCat);
  fd.append('algorithm', document.getElementById('train-algo').value);
  fd.append('epochs', document.getElementById('train-epochs').value);
  fd.append('lr', document.getElementById('train-lr').value);
  fd.append('test_size', document.getElementById('train-test').value);

  const r = await fetch('/api/train', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { toast(j.detail || '启动失败', false); return; }

  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('metric-cards').style.display = 'none';
  if (!lossChart) lossChart = echarts.init(document.getElementById('loss-chart'));
  lossChart.clear();

  if (trainTimer) clearInterval(trainTimer);
  trainTimer = setInterval(pollTrain, 500);
}

async function pollTrain() {
  const s = await fetch('/api/train/progress').then(x => x.json());
  document.getElementById('progress-fill').style.width = s.progress + '%';
  document.getElementById('progress-text').textContent = s.message || `训练中 ${s.progress}%`;

  if (s.loss_curve && s.loss_curve.length) {
    lossChart.setOption({
      title: { text: '损失值变化曲线（测试集 MSE）', textStyle: { fontSize: 13 } },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', name: '训练轮数', data: s.loss_curve.map(p => p.epoch) },
      yAxis: { type: 'value', name: '损失值' },
      grid: { left: 70, right: 30, bottom: 50 },
      series: [{ type: 'line', data: s.loss_curve.map(p => p.loss), smooth: true,
                 itemStyle: { color: '#2F78FF' }, areaStyle: { opacity: 0.08 } }]
    });
  }

  if (s.status === 'done') {
    clearInterval(trainTimer); trainTimer = null;
    document.getElementById('metric-cards').style.display = 'flex';
    document.getElementById('m-mse').textContent = s.metrics.mse;
    document.getElementById('m-mae').textContent = s.metrics.mae;
    document.getElementById('m-r2').textContent = s.metrics.r2;
    document.getElementById('m-acc').textContent = (s.metrics.accuracy * 100).toFixed(1) + '%';
    toast('训练完成');
    loadStats();
  } else if (s.status === 'error') {
    clearInterval(trainTimer); trainTimer = null;
    toast('训练失败：' + s.message, false);
  }
}

async function refreshOptimize() {
  const j = await fetch('/api/models').then(x => x.json());
  const sel = document.getElementById('opt-model');
  sel.innerHTML = '';
  j.models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = `${m.algo_label} · ${m.dataset}（R²=${m.metrics.r2}）`;
    sel.appendChild(o);
  });
  if (j.models.length) loadOptInputs();
}

async function loadOptInputs() {
  const modelId = document.getElementById('opt-model').value;
  if (!modelId) { toast('请先在「模型训练」页训练模型', false); return; }
  const models = await fetch('/api/models').then(x => x.json());
  const m = models.models.find(x => x.id === modelId);
  const info = await fetch(`/api/params?dataset=${encodeURIComponent(m.dataset)}`).then(x => x.json());

  const box = document.getElementById('opt-inputs');
  box.innerHTML = '';
  info.features.forEach(f => {
    const b = info.bounds[f];
    const row = document.createElement('div');
    row.className = 'opt-input-row';
    row.innerHTML = `<label title="${f}">${f}</label>
      <input type="number" step="any" data-param="${f}" value="${b.mean}">`;
    box.appendChild(row);
  });
}
document.getElementById('opt-model').addEventListener('change', loadOptInputs);

function fillCurrentParams() {
  document.querySelectorAll('#opt-inputs input').forEach(i => i.focus());
  toast('请直接在各参数输入框中填写当前值');
}

async function runOptimize() {
  const modelId = document.getElementById('opt-model').value;
  if (!modelId) { toast('请先训练模型', false); return; }
  const params = {};
  document.querySelectorAll('#opt-inputs input').forEach(i => {
    params[i.dataset.param] = parseFloat(i.value) || 0;
  });
  const fd = new FormData();
  fd.append('model_id', modelId);
  fd.append('params', JSON.stringify(params));

  const r = await fetch('/api/optimize', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { toast(j.detail || '优化失败', false); return; }

  document.getElementById('opt-summary').textContent =
    `预测${j.target}：${j.current_pred} → ${j.optimized_pred}（提升 ${j.improvement}%）`;

  if (!optChart) optChart = echarts.init(document.getElementById('opt-chart'));
  optChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['优化前', '优化后'] },
    xAxis: { type: 'category', data: j.chart.names, axisLabel: { rotate: 40, fontSize: 10 } },
    yAxis: { type: 'value', max: 1 },
    grid: { left: 50, right: 20, bottom: 80 },
    series: [
      { name: '优化前', type: 'bar', data: j.chart.before, itemStyle: { color: '#C0C4CC' } },
      { name: '优化后', type: 'bar', data: j.chart.after, itemStyle: { color: '#2F78FF' } }
    ]
  });

  const table = document.getElementById('opt-table');
  let html = '<thead><tr><th>参数名称</th><th>当前值</th><th>推荐值</th><th>调整方向</th></tr></thead><tbody>';
  j.params.forEach(p => {
    html += `<tr><td>${p.param}</td><td>${p.current}</td>
      <td class="rec">${p.recommended}</td><td>${p.direction}</td></tr>`;
  });
  table.innerHTML = html + '</tbody>';
  toast('优化完成');
  loadStats();
}

function exportResult() { window.location.href = '/api/export'; }

loadStats();
refreshDatasets();
