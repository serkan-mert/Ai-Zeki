// ---------- Translation System ----------
const LANG = {
  en: {
    'nav.overview': 'Overview', 'nav.create': 'Create Model', 'nav.layers': 'Add Layers',
    'nav.inspect': 'Inspect Model', 'nav.predict': 'Predict', 'nav.train': 'Train',
    'nav.evaluate': 'Evaluate', 'nav.files': 'Save / Load', 'nav.resources': 'Resources', 'nav.settings': 'Settings',

    'status.connected': 'Connected to Zeki (SSE)', 'status.disconnected': 'Disconnected', 'status.noModel': 'No model loaded', 'status.modelLoaded': 'Model loaded',

    'overview.title': 'Model Overview',
    'overview.placeholder': 'Click <strong>Refresh</strong> to load model info',
    'overview.refresh': 'Refresh', 'overview.listLayers': 'List Layers',

    'create.title': 'Create New Model',
    'create.inputShape': 'Input Shape (comma-separated, e.g. <code>1, 784</code>)',
    'create.createBtn': 'Create Model',

    'layers.title': 'Add Layers', 'layers.layerType': 'Layer Type', 'layers.layerName': 'Layer Name (optional)',
    'layers.addBtn': 'Add Layer',

    'inspect.title': 'Inspect Model', 'inspect.getInfo': 'Get Model Info',
    'inspect.listAll': 'List All Layers', 'inspect.getOutput': 'Get Layer Output', 'inspect.layerIndex': 'Layer Index',

    'predict.title': 'Run Prediction', 'predict.tab1': 'Standard', 'predict.tab2': 'Auto JSON', 'predict.tab3': 'Tokenization',
    'predict.inputData': 'Input Data (comma-separated floats)',
    'predict.inputShape': 'Input Shape', 'predict.predictBtn': 'Predict',
    'predict.inputDataRaw': 'Raw Input Data (nested JSON, shape auto-detected)',
    'predict.textData': 'Text Input Data (one sentence per line)',
    'predict.textMaxSeqLen': 'Max Sequence Length',

    'train.title': 'Train Model', 'train.tab1': 'Standard', 'train.tab2': 'Auto JSON', 'train.tab3': 'Tokenization',
    'train.inputData': 'Train Input Data (flat JSON array)',
    'train.targetData': 'Train Target Data (flat JSON array)',
    'train.inputDataRaw': 'Raw Train Input Data (nested JSON, shape auto-detected)',
    'train.targetDataRaw': 'Raw Train Target Data (nested JSON, shape auto-detected)',
    'train.textData': 'Text Input Data (one sentence per line)',
    'train.textTargets': 'Text Target Data (one target text per line)',
    'train.textMaxSeqLen': 'Input Max Sequence Length',
    'train.textTargetMaxSeqLen': 'Target Max Sequence Length',
    'train.inputShape': 'Input Shape', 'train.targetShape': 'Target Shape',
    'train.epochs': 'Epochs', 'train.batchSize': 'Batch Size',
    'train.trainBtn': 'Train', 'train.stepBtn': 'Single Train Step',

    'eval.title': 'Evaluate Model', 'eval.tab1': 'Standard', 'eval.tab2': 'Auto JSON', 'eval.tab3': 'Tokenization',
    'eval.inputData': 'Test Input Data (flat JSON array)',
    'eval.targetData': 'Test Target Data (flat JSON array)',
    'eval.inputDataRaw': 'Raw Input Data (nested JSON, shape auto-detected)',
    'eval.targetDataRaw': 'Raw Target Data (nested JSON, shape auto-detected)',
    'eval.textData': 'Text Input Data (one sentence per line)',
    'eval.textTargets': 'Text Target Data (one target text per line)',
    'eval.textMaxSeqLen': 'Input Max Sequence Length',
    'eval.textTargetMaxSeqLen': 'Target Max Sequence Length',
    'eval.inputShape': 'Input Shape', 'eval.targetShape': 'Target Shape', 'eval.evaluateBtn': 'Evaluate',

    'files.title': 'Save & Load Model', 'files.filename': 'Filename',
    'files.saveBtn': 'Save Model', 'files.loadBtn': 'Load Model', 'files.exitBtn': 'Exit Model',
    'files.exitConfirm': 'Do you want to save the model before exiting?',
    'files.exitSave': 'Save and Exit', 'files.exitNosave': 'Exit without Saving', 'files.exitCancel': 'Cancel',
    'files.exitPrompt': 'Enter filename to save:',
    'files.exitNoName': 'No filename entered. Auto-generating...',
    'files.exitDone': 'Model exited successfully',
    'files.exitSaved': 'Model saved and exited',

    'resources.title': 'MCP Resources', 'resources.uri': 'Resource URI', 'resources.readBtn': 'Read Resource',
    'resources.prompts': 'Prompts', 'resources.promptName': 'Prompt Name', 'resources.getPromptBtn': 'Get Prompt',

    'settings.title': 'Settings', 'settings.trainingMode': 'Training Mode',
    'settings.evalMode': 'Evaluation', 'settings.trainMode': 'Training', 'settings.applyBtn': 'Apply',
    'settings.presetTitle': 'Preset: Create Simple Classifier',
    'settings.presetDesc': 'Creates a 3-layer model: Dense(5\u219210) \u2192 ReLU \u2192 Dense(10\u21923) \u2192 Softmax',
    'settings.presetBtn': 'Create Classifier',

    'layerType.dense': 'Dense', 'layerType.conv2d': 'Conv2D', 'layerType.relu': 'ReLU',
    'layerType.sigmoid': 'Sigmoid', 'layerType.tanh': 'Tanh', 'layerType.softmax': 'Softmax',
    'layerType.dropout': 'Dropout', 'layerType.flatten': 'Flatten', 'layerType.lstm': 'LSTM',
    'layerType.gru': 'GRU', 'layerType.batchnorm': 'BatchNorm', 'layerType.attention': 'Attention',
    'layerType.transformer': 'Transformer',

    'layerParams.input_dim': 'Input Dim', 'layerParams.output_dim': 'Output Dim',
    'layerParams.in_channels': 'In Channels', 'layerParams.out_channels': 'Out Channels',
    'layerParams.kernel_size': 'Kernel Size', 'layerParams.stride': 'Stride', 'layerParams.padding': 'Padding',
    'layerParams.rate': 'Dropout Rate (0-1)',
    'layerParams.hidden_dim': 'Hidden Dim', 'layerParams.num_heads': 'Num Heads',
    'layerParams.dim': 'Dim', 'layerParams.embed_dim': 'Embed Dim', 'layerParams.ff_dim': 'FF Dim',

    'table.idx': '#', 'table.type': 'Type', 'table.name': 'Name', 'table.input': 'Input', 'table.output': 'Output',
    'table.delete': 'Delete',

    'preset.create': 'Create', 'preset.dense': 'Dense',
    'lang.en': 'English', 'lang.tr': 'T\u00fcrk\u00e7e',
    'lang.label': 'Language'
  },

  tr: {
    'nav.overview': 'Genel Bak\u0131\u015f', 'nav.create': 'Model Olu\u015ftur', 'nav.layers': 'Katman Ekle',
    'nav.inspect': 'Modeli \u0130ncele', 'nav.predict': 'Tahmin', 'nav.train': 'E\u011fit',
    'nav.evaluate': 'De\u011ferlendir', 'nav.files': 'Kaydet / Y\u00fckle', 'nav.resources': 'Kaynaklar', 'nav.settings': 'Ayarlar',

    'status.connected': 'Zeki\'ye ba\u011fland\u0131 (SSE)', 'status.disconnected': 'Ba\u011flant\u0131 kesildi', 'status.noModel': 'Model y\u00fckl\u00fc de\u011fil', 'status.modelLoaded': 'Model y\u00fcklendi',

    'overview.title': 'Model Genel Bak\u0131\u015f',
    'overview.placeholder': 'Model bilgisini y\u00fcklemek i\u00e7in <strong>Yenile</strong>\'ye t\u0131klay\u0131n',
    'overview.refresh': 'Yenile', 'overview.listLayers': 'Katmanlar\u0131 Listele',

    'create.title': 'Yeni Model Olu\u015ftur',
    'create.inputShape': 'Giri\u015f \u015eekli (virg\u00fclle ayr\u0131lm\u0131\u015f, \u00f6rn. <code>1, 784</code>)',
    'create.createBtn': 'Model Olu\u015ftur',

    'layers.title': 'Katman Ekle', 'layers.layerType': 'Katman T\u00fcr\u00fc', 'layers.layerName': 'Katman Ad\u0131 (iste\u011fe ba\u011fl\u0131)',
    'layers.addBtn': 'Katman Ekle',

    'inspect.title': 'Modeli \u0130ncele', 'inspect.getInfo': 'Model Bilgisi Al',
    'inspect.listAll': 'T\u00fcm Katmanlar\u0131 Listele', 'inspect.getOutput': 'Katman \u00c7\u0131kt\u0131s\u0131n\u0131 Al',
    'inspect.layerIndex': 'Katman \u0130ndeksi',

    'predict.title': 'Tahmin \u00c7al\u0131\u015ft\u0131r', 'predict.tab1': 'Standart', 'predict.tab2': 'Otomatik JSON', 'predict.tab3': 'Tokenizasyon',
    'predict.inputData': 'Giri\u015f Verileri (virg\u00fclle ayr\u0131lm\u0131\u015f say\u0131lar)',
    'predict.inputShape': 'Giri\u015f \u015eekli', 'predict.predictBtn': 'Tahmin Et',
    'predict.inputDataRaw': 'Ham Giri\u015f Verileri (i\u00e7 i\u00e7e JSON, \u015fekil otomatik)',
    'predict.textData': 'Metin Giri\u015f Verileri (her sat\u0131r bir c\u00fcmle)',
    'predict.textMaxSeqLen': 'Maks. Dizi Uzunlu\u011fu',

    'train.title': 'Modeli E\u011fit', 'train.tab1': 'Standart', 'train.tab2': 'Otomatik JSON', 'train.tab3': 'Tokenizasyon',
    'train.inputData': 'E\u011fitim Giri\u015f Verileri (d\u00fcz JSON dizi)',
    'train.targetData': 'E\u011fitim Hedef Verileri (d\u00fcz JSON dizi)',
    'train.inputDataRaw': 'Ham E\u011fitim Giri\u015f Verileri (i\u00e7 i\u00e7e JSON, \u015fekil otomatik)',
    'train.targetDataRaw': 'Ham E\u011fitim Hedef Verileri (i\u00e7 i\u00e7e JSON, \u015fekil otomatik)',
    'train.textData': 'Metin Giri\u015f Verileri (her sat\u0131r bir c\u00fcmle)',
    'train.textTargets': 'Metin Hedef Verileri (her sat\u0131r bir hedef metin)',
    'train.textMaxSeqLen': 'Giri\u015f Maks. Dizi Uzunlu\u011fu',
    'train.textTargetMaxSeqLen': 'Hedef Maks. Dizi Uzunlu\u011fu',
    'train.inputShape': 'Giri\u015f \u015eekli', 'train.targetShape': 'Hedef \u015eekli',
    'train.epochs': 'Epok Say\u0131s\u0131', 'train.batchSize': 'Parti Boyutu',
    'train.trainBtn': 'E\u011fit', 'train.stepBtn': 'Tek Ad\u0131m E\u011fit',

    'eval.title': 'Modeli De\u011ferlendir', 'eval.tab1': 'Standart', 'eval.tab2': 'Otomatik JSON', 'eval.tab3': 'Tokenizasyon',
    'eval.inputData': 'Test Giri\u015f Verileri (d\u00fcz JSON dizi)',
    'eval.targetData': 'Test Hedef Verileri (d\u00fcz JSON dizi)',
    'eval.inputDataRaw': 'Ham Giri\u015f Verileri (i\u00e7 i\u00e7e JSON, \u015fekil otomatik)',
    'eval.targetDataRaw': 'Ham Hedef Verileri (i\u00e7 i\u00e7e JSON, \u015fekil otomatik)',
    'eval.textData': 'Metin Giri\u015f Verileri (her sat\u0131r bir c\u00fcmle)',
    'eval.textTargets': 'Metin Hedef Verileri (her sat\u0131r bir hedef metin)',
    'eval.textMaxSeqLen': 'Giri\u015f Maks. Dizi Uzunlu\u011fu',
    'eval.textTargetMaxSeqLen': 'Hedef Maks. Dizi Uzunlu\u011fu',
    'eval.inputShape': 'Giri\u015f \u015eekli', 'eval.targetShape': 'Hedef \u015eekli', 'eval.evaluateBtn': 'De\u011ferlendir',

    'files.title': 'Model Kaydet / Y\u00fckle', 'files.filename': 'Dosya Ad\u0131',
    'files.saveBtn': 'Modeli Kaydet', 'files.loadBtn': 'Modeli Y\u00fckle', 'files.exitBtn': 'Modelden \u00c7\u0131k',
    'files.exitConfirm': '\u00c7\u0131kmadan \u00f6nce model kaydedilsin mi?',
    'files.exitSave': 'Kaydet ve \u00c7\u0131k', 'files.exitNosave': 'Kaydetmeden \u00c7\u0131k', 'files.exitCancel': '\u0130ptal',
    'files.exitPrompt': 'Kaydetmek i\u00e7in dosya ad\u0131 girin:',
    'files.exitNoName': 'Dosya ad\u0131 girilmedi. Otomatik olu\u015fturuluyor...',
    'files.exitDone': 'Modelden \u00e7\u0131k\u0131ld\u0131',
    'files.exitSaved': 'Model kaydedildi ve \u00e7\u0131k\u0131ld\u0131',

    'resources.title': 'MCP Kaynaklar\u0131', 'resources.uri': 'Kaynak URI', 'resources.readBtn': 'Kayna\u011f\u0131 Oku',
    'resources.prompts': 'Komut \u015eablonlar\u0131', 'resources.promptName': 'Komut Ad\u0131', 'resources.getPromptBtn': 'Komutu Al',

    'settings.title': 'Ayarlar', 'settings.trainingMode': 'E\u011fitim Modu',
    'settings.evalMode': 'De\u011ferlendirme', 'settings.trainMode': 'E\u011fitim', 'settings.applyBtn': 'Uygula',
    'settings.presetTitle': 'Haz\u0131r: Basit S\u0131n\u0131fland\u0131r\u0131c\u0131 Olu\u015ftur',
    'settings.presetDesc': '3 katmanl\u0131 model olu\u015fturur: Dense(5\u219210) \u2192 ReLU \u2192 Dense(10\u21923) \u2192 Softmax',
    'settings.presetBtn': 'S\u0131n\u0131fland\u0131r\u0131c\u0131 Olu\u015ftur',

    'layerType.dense': 'Dense', 'layerType.conv2d': 'Conv2D', 'layerType.relu': 'ReLU',
    'layerType.sigmoid': 'Sigmoid', 'layerType.tanh': 'Tanh', 'layerType.softmax': 'Softmax',
    'layerType.dropout': 'Dropout', 'layerType.flatten': 'Flatten', 'layerType.lstm': 'LSTM',
    'layerType.gru': 'GRU', 'layerType.batchnorm': 'BatchNorm', 'layerType.attention': 'Attention',
    'layerType.transformer': 'Transformer',

    'layerParams.input_dim': 'Giri\u015f Boyutu', 'layerParams.output_dim': '\u00c7\u0131k\u0131\u015f Boyutu',
    'layerParams.in_channels': 'Giri\u015f Kanallar\u0131', 'layerParams.out_channels': '\u00c7\u0131k\u0131\u015f Kanallar\u0131',
    'layerParams.kernel_size': 'Kernel Boyutu', 'layerParams.stride': 'Ad\u0131m', 'layerParams.padding': 'Dolgu',
    'layerParams.rate': 'Dropout Oran\u0131 (0-1)',
    'layerParams.hidden_dim': 'Gizli Boyut', 'layerParams.num_heads': 'Ba\u015f Say\u0131s\u0131',
    'layerParams.dim': 'Boyut', 'layerParams.embed_dim': 'G\u00f6mme Boyutu', 'layerParams.ff_dim': 'FF Boyutu',

    'table.idx': '#', 'table.type': 'T\u00fcr', 'table.name': 'Ad', 'table.input': 'Giri\u015f', 'table.output': '\u00c7\u0131k\u0131\u015f',
    'table.delete': 'Sil',

    'preset.create': 'Olu\u015ftur', 'preset.dense': 'Dense',
    'lang.en': 'English', 'lang.tr': 'T\u00fcrk\u00e7e',
    'lang.label': 'Dil'
  }
};

let currentLang = localStorage.getItem('zeki_lang') || 'en';

function t(key) {
  return (LANG[currentLang] && LANG[currentLang][key]) || (LANG.en[key]) || key;
}

function applyLang() {
  document.documentElement.lang = currentLang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (el.tagName === 'OPTION') {
      el.textContent = t(key);
    } else if (el.dataset.i18nHtml) {
      el.innerHTML = t(key);
    } else {
      el.textContent = t(key);
    }
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    el.title = t(el.dataset.i18nTitle);
  });
  localStorage.setItem('zeki_lang', currentLang);
  updateLayerParams();
  if (typeof updateStatusText === 'function') updateStatusText();
  if (typeof updateModelStatus === 'function') updateModelStatus();
}

function setLang(lang) {
  currentLang = lang;
  applyLang();
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
}

// ---------- Navigation ----------
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('section-' + btn.dataset.section).classList.add('active');
  });
});

// ---------- Connection Status ----------
function updateStatusText() {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  if (!txt) return;
  if (txt.dataset.connected === 'true') {
    dot.className = 'status-dot connected';
    txt.textContent = t('status.connected');
  } else {
    dot.className = 'status-dot';
    txt.textContent = t('status.disconnected');
  }
}

async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (s.connected) {
      dot.className = 'status-dot connected';
      txt.dataset.connected = 'true';
      txt.textContent = t('status.connected');
    } else {
      dot.className = 'status-dot';
      txt.dataset.connected = 'false';
      txt.textContent = t('status.disconnected');
    }
  } catch { }
}
setInterval(checkStatus, 3000);
checkStatus();

// ---------- Generic MCP call ----------
async function mcpCall(tool, params = {}) {
  const r = await fetch('/api/mcp/' + tool, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return r.json();
}

function showResult(el, data) {
  const isErr = data && (data.error || data.isError);
  el.className = 'result mt-1' + (isErr ? ' error' : ' success');
  el.textContent = JSON.stringify(data, null, 2);
}

// ---------- Model Status ----------
async function updateModelStatus() {
  const el = document.getElementById('modelInfo');
  const r = await mcpCall('get_model_info');
  if (r.error || r.isError) {
    el.textContent = t('status.noModel');
  } else {
    el.textContent = t('status.modelLoaded');
  }
}

// ---------- Overview ----------
async function refreshModelInfo() {
  const el = document.getElementById('overviewCard');
  const r = await mcpCall('get_model_info');
  el.innerHTML = '<pre class="result success">' + JSON.stringify(r, null, 2) + '</pre>';
  updateModelStatus();
}

async function listLayers() {
  const el = document.getElementById('overviewLayers');
  const r = await mcpCall('list_layers');
  if (r.error || r.isError) { el.innerHTML = '<div class="result error">' + JSON.stringify(r, null, 2) + '</div>'; return; }
  const layers = r.result && r.result.content ? JSON.parse(r.result.content[0].text) : (r.content || []);
  if (!Array.isArray(layers)) { el.innerHTML = '<div class="result">' + JSON.stringify(r, null, 2) + '</div>'; return; }

  const ignoreKeys = new Set(['index', 'params', 'type', 'name']);
  const allKeys = new Set(['type', 'name']);
  layers.forEach(l => { Object.keys(l).forEach(k => { if (!ignoreKeys.has(k)) allKeys.add(k); }); });
  const keys = Array.from(allKeys);

  let html = '<table class="layer-table" style="font-size:13px"><tr><th>#</th>';
  keys.forEach(k => { html += '<th>' + k + '</th>'; });
  html += '<th>Params</th><th>' + t('table.delete') + '</th></tr>';

  layers.forEach((l, i) => {
    html += '<tr><td>' + i + '</td>';
    keys.forEach(k => {
      let v = l[k];
      if (v === undefined || v === null) v = '-';
      else if (Array.isArray(v)) v = '[' + v.join(',') + ']';
      else if (typeof v === 'object') v = JSON.stringify(v);
      html += '<td>' + v + '</td>';
    });
    if (l.params && Object.keys(l.params).length > 0) {
      let phtml = '';
      for (const [pk, pv] of Object.entries(l.params)) {
        phtml += '<span class="param-tag">' + pk + ': ' + pv + '</span> ';
      }
      html += '<td style="white-space:normal">' + phtml + '</td>';
    } else {
      html += '<td>-</td>';
    }
    if (i > 0) {
      html += '<td><button class="btn-del" onclick="removeLayer(' + i + ')">' + t('table.delete') + '</button></td>';
    } else {
      html += '<td>-</td>';
    }
    html += '</tr>';
  });
  html += '</table>';
  el.innerHTML = html;
}

async function removeLayer(index) {
  const r = await mcpCall('remove_layer', { layer_index: index });
  if (r.error || r.isError) {
    alert('Error: ' + (r.error?.message || JSON.stringify(r)));
    return;
  }
  listLayers();
}

// ---------- Create Model ----------
async function createModel() {
  const val = document.getElementById('createShape').value;
  const shape = val.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const r = await mcpCall('create_model', { input_shape: shape });
  showResult(document.getElementById('createResult'), r);
  updateModelStatus();
}

// ---------- Add Layers ----------
const layerParamConfig = {
  dense: { fields: ['input_dim', 'output_dim'] },
  conv2d: { fields: ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'] },
  relu: { fields: [] }, sigmoid: { fields: [] }, tanh: { fields: [] }, softmax: { fields: [] },
  flatten: { fields: [] },
  dropout: { fields: ['rate'] },
  lstm: { fields: ['input_dim', 'hidden_dim'] },
  gru: { fields: ['input_dim', 'hidden_dim'] },
  batchnorm: { fields: ['dim'] },
  attention: { fields: ['embed_dim'] },
  transformer: { fields: ['embed_dim', 'num_heads', 'ff_dim'] }
};

function updateLayerParams() {
  const type = document.getElementById('layerType').value;
  const cfg = layerParamConfig[type] || layerParamConfig.dense;
  const container = document.getElementById('layerParams');
  container.innerHTML = '';
  cfg.fields.forEach(f => {
    const lbl = document.createElement('label');
    lbl.textContent = t('layerParams.' + f);
    lbl.htmlFor = 'lp_' + f;
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.id = 'lp_' + f;
    inp.placeholder = t('layerParams.' + f);
    inp.value = f === 'input_dim' || f === 'in_channels' || f === 'dim' ? '1' : f === 'output_dim' || f === 'out_channels' ? '10' : f === 'kernel_size' ? '3' : f === 'rate' ? '0.5' : f === 'hidden_dim' ? '10' : f === 'num_heads' ? '4' : f === 'embed_dim' ? '64' : f === 'ff_dim' ? '256' : f === 'stride' || f === 'padding' ? '1' : '';
    container.appendChild(lbl);
    container.appendChild(inp);
  });
}

async function addLayer() {
  const type = document.getElementById('layerType').value;
  const name = document.getElementById('layerName').value.trim() || undefined;
  const cfg = layerParamConfig[type] || layerParamConfig.dense;
  const layerParams = {};
  cfg.fields.forEach(f => {
    const v = document.getElementById('lp_' + f);
    if (v) layerParams[f] = parseFloat(v.value);
  });
  const args = { layer_type: type };
  if (name) args.name = name;
  if (cfg.fields.length > 0) args.params = layerParams;
  const r = await mcpCall('add_layer', args);
  showResult(document.getElementById('layerResult'), r);
}

// ---------- Inspect ----------
async function getModelInfo() {
  const r = await mcpCall('get_model_info');
  showResult(document.getElementById('inspectResult'), r);
}

async function listLayersDetailed() {
  const r = await mcpCall('list_layers');
  showResult(document.getElementById('inspectResult'), r);
}

async function getLayerOutput() {
  const idx = parseInt(document.getElementById('layerIndex').value);
  const r = await mcpCall('get_layer_output', { layer_index: idx });
  showResult(document.getElementById('inspectResult'), r);
}

// ---------- Predict ----------
async function runPredict() {
  const data = document.getElementById('predictData').value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
  const shapeVal = document.getElementById('predictShape').value;
  const shape = shapeVal.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const r = await mcpCall('predict', { input_data: data, input_shape: shape });
  showResult(document.getElementById('predictResult'), r);
}

// ---------- Predict - Auto JSON & Tokenization tabs ----------
function switchPredictTab(tabId) {
  document.querySelectorAll('#section-predict .tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('#section-predict .tab-content').forEach(c => c.classList.toggle('active', c.id === tabId));
  if (tabId === 'predict-auto') updatePredictAutoShape();
  if (tabId === 'predict-token') updatePredictTokenInfo();
}

function updatePredictAutoShape() {
  const el = document.getElementById('predictInputRaw');
  const info = document.getElementById('predictAutoShape');
  try {
    const parsed = JSON.parse(el.value);
    const shape = detectShape(parsed);
    info.textContent = 'Detected shape: [' + shape.join(',') + ']';
    info.style.color = '#7ee787';
  } catch {
    info.textContent = 'Invalid JSON';
    info.style.color = '#ff7b72';
  }
}

document.getElementById('predictInputRaw').addEventListener('input', updatePredictAutoShape);

async function runPredictAuto() {
  const parsed = JSON.parse(document.getElementById('predictInputRaw').value);
  const flat = flattenArray(parsed);
  const shape = detectShape(parsed);
  const r = await mcpCall('predict', { input_data: flat, input_shape: shape });
  showResult(document.getElementById('predictAutoResult'), r);
}

function updatePredictTokenInfo() {
  const text = document.getElementById('predictTextInput').value;
  const maxLen = parseInt(document.getElementById('predictMaxSeqLen').value) || 1;
  const lines = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const info = document.getElementById('predictTokenInfo');
  if (lines.length === 0) { info.textContent = 'No input text'; return; }
  const vocab = buildVocab(lines);
  const shape = [lines.length, maxLen];
  info.textContent = 'Vocab: ' + Object.keys(vocab).length + ' words  Shape: [' + shape.join(',') + ']';
}

['predictTextInput', 'predictMaxSeqLen'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updatePredictTokenInfo);
});

async function runPredictToken() {
  const text = document.getElementById('predictTextInput').value;
  const maxLen = parseInt(document.getElementById('predictMaxSeqLen').value) || 1;
  const sentences = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  if (sentences.length === 0) { alert('No input text'); return; }
  const vocab = buildVocab(sentences);
  const seqs = sentences.map(s => tokenizeSentence(s, vocab, maxLen));
  const flat = seqs.flat();
  const shape = [sentences.length, maxLen];
  const r = await mcpCall('predict', { input_data: flat, input_shape: shape });
  showResult(document.getElementById('predictTokenResult'), r);
}

// ---------- Train ----------
function parseJsonArray(str) {
  try { return JSON.parse(str); } catch { return JSON.parse('[' + str + ']'); }
}

function flattenArray(arr) {
  const result = [];
  function walk(v) {
    if (Array.isArray(v)) v.forEach(walk);
    else if (typeof v === 'number') result.push(v);
  }
  walk(arr);
  return result;
}

async function trainModel() {
  const trainInput = flattenArray(parseJsonArray(document.getElementById('trainInput').value));
  const trainTarget = flattenArray(parseJsonArray(document.getElementById('trainTarget').value));
  const inputShape = document.getElementById('trainInputShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const targetShape = document.getElementById('trainTargetShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const epochs = parseInt(document.getElementById('trainEpochs').value);
  const batchSize = parseInt(document.getElementById('trainBatchSize').value);
  const r = await mcpCall('train', {
    train_input: trainInput, train_target: trainTarget,
    input_shape: inputShape, target_shape: targetShape,
    epochs, batch_size: batchSize
  });
  showResult(document.getElementById('trainResult'), r);
}

async function trainStep() {
  const trainInput = flattenArray(parseJsonArray(document.getElementById('trainInput').value));
  const trainTarget = flattenArray(parseJsonArray(document.getElementById('trainTarget').value));
  const inputShape = document.getElementById('trainInputShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const targetShape = document.getElementById('trainTargetShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const r = await mcpCall('train_step', {
    input_data: trainInput, target_data: trainTarget,
    input_shape: inputShape, target_shape: targetShape
  });
  showResult(document.getElementById('trainResult'), r);
}

// ---------- Train - Auto JSON Tab ----------
function detectShape(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return [];
  const shape = [arr.length];
  if (Array.isArray(arr[0])) {
    const rest = detectShape(arr[0]);
    shape.push(...rest);
  }
  return shape;
}

function switchTrainTab(tabId) {
  document.querySelectorAll('#section-train .tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('#section-train .tab-content').forEach(c => c.classList.toggle('active', c.id === tabId));
  if (tabId === 'train-auto') updateAutoShapes();
}

function updateAutoShapes() {
  const inputEl = document.getElementById('trainInputRaw');
  const targetEl = document.getElementById('trainTargetRaw');
  const infoEl = document.getElementById('trainAutoShape');
  try {
    const inputParsed = JSON.parse(inputEl.value);
    const targetParsed = JSON.parse(targetEl.value);
    const inputShape = detectShape(inputParsed);
    const targetShape = detectShape(targetParsed);
    infoEl.textContent = 'Detected shapes - Input: [' + inputShape.join(',') + ']  Target: [' + targetShape.join(',') + ']';
  } catch {
    infoEl.textContent = 'Invalid JSON';
    infoEl.style.color = '#ff7b72';
    return;
  }
  infoEl.style.color = '#7ee787';
}

['trainInputRaw', 'trainTargetRaw'].forEach(id => {
  document.getElementById(id).addEventListener('input', updateAutoShapes);
});

async function trainModelAuto() {
  const inputParsed = JSON.parse(document.getElementById('trainInputRaw').value);
  const targetParsed = JSON.parse(document.getElementById('trainTargetRaw').value);
  const inputFlat = flattenArray(inputParsed);
  const targetFlat = flattenArray(targetParsed);
  const inputShape = detectShape(inputParsed);
  const targetShape = detectShape(targetParsed);
  const epochs = parseInt(document.getElementById('trainEpochsAuto').value);
  const batchSize = parseInt(document.getElementById('trainBatchSizeAuto').value);
  const r = await mcpCall('train', {
    train_input: inputFlat, train_target: targetFlat,
    input_shape: inputShape, target_shape: targetShape,
    epochs, batch_size: batchSize
  });
  showResult(document.getElementById('trainAutoResult'), r);
}

async function trainStepAuto() {
  const inputParsed = JSON.parse(document.getElementById('trainInputRaw').value);
  const targetParsed = JSON.parse(document.getElementById('trainTargetRaw').value);
  const inputFlat = flattenArray(inputParsed);
  const targetFlat = flattenArray(targetParsed);
  const inputShape = detectShape(inputParsed);
  const targetShape = detectShape(targetParsed);
  const r = await mcpCall('train_step', {
    input_data: inputFlat, target_data: targetFlat,
    input_shape: inputShape, target_shape: targetShape
  });
  showResult(document.getElementById('trainAutoResult'), r);
}

// ---------- Train - Tokenization Tab ----------
function buildVocab(sentences) {
  const wordSet = new Set();
  sentences.forEach(s => {
    s.toLowerCase().split(/\s+/).filter(w => w.length > 0).forEach(w => wordSet.add(w));
  });
  const vocab = Array.from(wordSet);
  const word2idx = { '<PAD>': 0, '<UNK>': 1 };
  vocab.forEach((w, i) => { word2idx[w] = i + 2; });
  return word2idx;
}

function tokenizeSentence(sentence, word2idx, maxLen) {
  const words = sentence.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  const indices = words.map(w => word2idx[w] !== undefined ? word2idx[w] : word2idx['<UNK>']);
  while (indices.length < maxLen) indices.push(0);
  return indices.slice(0, maxLen);
}

function updateTokenInfo() {
  const text = document.getElementById('trainTextInput').value;
  const targetText = document.getElementById('trainTextTarget').value;
  const maxLen = parseInt(document.getElementById('trainMaxSeqLen').value) || 1;
  const targetMaxLen = parseInt(document.getElementById('trainTargetMaxSeqLen').value) || 1;
  const lines = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const targetLines = targetText.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const info = document.getElementById('trainTokenInfo');
  if (lines.length === 0) { info.textContent = 'No input text'; return; }
  const inVocab = buildVocab(lines);
  const tgtVocab = buildVocab(targetLines);
  const inShape = [lines.length, maxLen];
  const tgtShape = [targetLines.length, targetMaxLen];
  info.textContent = 'Input vocab: ' + Object.keys(inVocab).length + ' words  Target vocab: ' + Object.keys(tgtVocab).length + ' words  Shapes: [' + inShape.join(',') + '] -> [' + tgtShape.join(',') + ']';
}

['trainTextInput', 'trainMaxSeqLen', 'trainTextTarget', 'trainTargetMaxSeqLen'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateTokenInfo);
});

async function trainModelToken() {
  const text = document.getElementById('trainTextInput').value;
  const targetText = document.getElementById('trainTextTarget').value;
  const maxLen = parseInt(document.getElementById('trainMaxSeqLen').value) || 1;
  const targetMaxLen = parseInt(document.getElementById('trainTargetMaxSeqLen').value) || 1;

  const sentences = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const targetLines = targetText.split('\n').map(s => s.trim()).filter(s => s.length > 0);

  if (sentences.length === 0) { alert('No input text'); return; }
  if (targetLines.length === 0) { alert('No target text'); return; }
  if (sentences.length !== targetLines.length) { alert('Input and target must have same number of lines'); return; }

  const inVocab = buildVocab(sentences);
  const tgtVocab = buildVocab(targetLines);
  const inSeqs = sentences.map(s => tokenizeSentence(s, inVocab, maxLen));
  const tgtSeqs = targetLines.map(s => tokenizeSentence(s, tgtVocab, targetMaxLen));
  const inputFlat = inSeqs.flat();
  const targetFlat = tgtSeqs.flat();
  const inputShape = [sentences.length, maxLen];
  const targetShape = [targetLines.length, targetMaxLen];

  const epochs = parseInt(document.getElementById('trainEpochsToken').value);
  const batchSize = parseInt(document.getElementById('trainBatchSizeToken').value);

  const info = document.getElementById('trainTokenInfo');
  info.textContent = 'Training: input_vocab=' + Object.keys(inVocab).length + ' target_vocab=' + Object.keys(tgtVocab).length + ' input_shape=[' + inputShape.join(',') + '] target_shape=[' + targetShape.join(',') + ']';

  const r = await mcpCall('train', {
    train_input: inputFlat, train_target: targetFlat,
    input_shape: inputShape, target_shape: targetShape,
    epochs, batch_size: batchSize
  });
  showResult(document.getElementById('trainTokenResult'), r);
}

// ---------- Evaluate ----------
async function evaluateModel() {
  const inputData = flattenArray(parseJsonArray(document.getElementById('evalInput').value));
  const targetData = flattenArray(parseJsonArray(document.getElementById('evalTarget').value));
  const inputShape = document.getElementById('evalInputShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const targetShape = document.getElementById('evalTargetShape').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const r = await mcpCall('evaluate', {
    input_data: inputData, target_data: targetData,
    input_shape: inputShape, target_shape: targetShape
  });
  showResult(document.getElementById('evalResult'), r);
}

// ---------- Evaluate - Auto JSON & Tokenization tabs ----------
function switchEvalTab(tabId) {
  document.querySelectorAll('#section-evaluate .tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('#section-evaluate .tab-content').forEach(c => c.classList.toggle('active', c.id === tabId));
  if (tabId === 'eval-auto') updateEvalAutoShape();
  if (tabId === 'eval-token') updateEvalTokenInfo();
}

function updateEvalAutoShape() {
  const inputEl = document.getElementById('evalInputRaw');
  const targetEl = document.getElementById('evalTargetRaw');
  const info = document.getElementById('evalAutoShape');
  try {
    const inParsed = JSON.parse(inputEl.value);
    const tgtParsed = JSON.parse(targetEl.value);
    const inShape = detectShape(inParsed);
    const tgtShape = detectShape(tgtParsed);
    info.textContent = 'Detected shapes - Input: [' + inShape.join(',') + ']  Target: [' + tgtShape.join(',') + ']';
    info.style.color = '#7ee787';
  } catch {
    info.textContent = 'Invalid JSON';
    info.style.color = '#ff7b72';
  }
}

['evalInputRaw', 'evalTargetRaw'].forEach(id => {
  document.getElementById(id).addEventListener('input', updateEvalAutoShape);
});

async function evaluateModelAuto() {
  const inParsed = JSON.parse(document.getElementById('evalInputRaw').value);
  const tgtParsed = JSON.parse(document.getElementById('evalTargetRaw').value);
  const inFlat = flattenArray(inParsed);
  const tgtFlat = flattenArray(tgtParsed);
  const inShape = detectShape(inParsed);
  const tgtShape = detectShape(tgtParsed);
  const r = await mcpCall('evaluate', {
    input_data: inFlat, target_data: tgtFlat,
    input_shape: inShape, target_shape: tgtShape
  });
  showResult(document.getElementById('evalAutoResult'), r);
}

function updateEvalTokenInfo() {
  const text = document.getElementById('evalTextInput').value;
  const targetText = document.getElementById('evalTextTarget').value;
  const maxLen = parseInt(document.getElementById('evalMaxSeqLen').value) || 1;
  const targetMaxLen = parseInt(document.getElementById('evalTargetMaxSeqLen').value) || 1;
  const lines = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const targetLines = targetText.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const info = document.getElementById('evalTokenInfo');
  if (lines.length === 0) { info.textContent = 'No input text'; return; }
  const inVocab = buildVocab(lines);
  const tgtVocab = buildVocab(targetLines);
  const inShape = [lines.length, maxLen];
  const tgtShape = [targetLines.length, targetMaxLen];
  info.textContent = 'Input vocab: ' + Object.keys(inVocab).length + ' words  Target vocab: ' + Object.keys(tgtVocab).length + ' words  Shapes: [' + inShape.join(',') + '] -> [' + tgtShape.join(',') + ']';
}

['evalTextInput', 'evalMaxSeqLen', 'evalTextTarget', 'evalTargetMaxSeqLen'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', updateEvalTokenInfo);
});

async function evaluateModelToken() {
  const text = document.getElementById('evalTextInput').value;
  const targetText = document.getElementById('evalTextTarget').value;
  const maxLen = parseInt(document.getElementById('evalMaxSeqLen').value) || 1;
  const targetMaxLen = parseInt(document.getElementById('evalTargetMaxSeqLen').value) || 1;

  const sentences = text.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  const targetLines = targetText.split('\n').map(s => s.trim()).filter(s => s.length > 0);

  if (sentences.length === 0) { alert('No input text'); return; }
  if (targetLines.length === 0) { alert('No target text'); return; }
  if (sentences.length !== targetLines.length) { alert('Input and target must have same number of lines'); return; }

  const inVocab = buildVocab(sentences);
  const tgtVocab = buildVocab(targetLines);
  const inSeqs = sentences.map(s => tokenizeSentence(s, inVocab, maxLen));
  const tgtSeqs = targetLines.map(s => tokenizeSentence(s, tgtVocab, targetMaxLen));
  const inFlat = inSeqs.flat();
  const tgtFlat = tgtSeqs.flat();
  const inShape = [sentences.length, maxLen];
  const tgtShape = [targetLines.length, targetMaxLen];

  const r = await mcpCall('evaluate', {
    input_data: inFlat, target_data: tgtFlat,
    input_shape: inShape, target_shape: tgtShape
  });
  showResult(document.getElementById('evalTokenResult'), r);
}

// ---------- Save / Load ----------
async function saveModel() {
  const filename = document.getElementById('fileFilename').value.trim();
  const r = await mcpCall('save_model', { filename });
  showResult(document.getElementById('fileResult'), r);
}

async function loadModel() {
  const filename = document.getElementById('fileFilename').value.trim();
  const r = await mcpCall('load_model', { filename });
  showResult(document.getElementById('fileResult'), r);
  updateModelStatus();
}

async function exitModel() {
  const el = document.getElementById('fileResult');
  const fin = document.getElementById('fileFilename');
  const hasModel = !(await mcpCall('get_model_info')).error;

  if (hasModel) {
    const save = await showConfirmModal(t('files.exitConfirm'));
    if (save) {
      let filename = fin.value.trim();
      if (!filename) {
        const now = new Date();
        const pad = n => String(n).padStart(2, '0');
        filename = 'model_' + now.getFullYear() + pad(now.getMonth()+1) + pad(now.getDate()) +
                   '_' + pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds()) + '.bin';
        fin.value = filename;
      }
      const r = await mcpCall('save_model', { filename });
      if (!r.error && !r.isError) {
        el.textContent = t('files.exitSaved') + ': ' + filename;
      } else {
        el.textContent = 'Save failed: ' + (r.error?.message || JSON.stringify(r));
      }
    } else {
      el.textContent = t('files.exitDone');
    }
  }

  await mcpCall('destroy_model', {});
  el.className = 'result mt-1 success';
  el.textContent = el.textContent || t('files.exitDone');
  document.getElementById('modelInfo').textContent = t('status.noModel');
}

function showConfirmModal(text) {
  return new Promise(resolve => {
    const overlay = document.getElementById('confirmModal');
    const msg = document.getElementById('confirmModalText');
    const yesBtn = document.getElementById('confirmYesBtn');
    const noBtn = document.getElementById('confirmNoBtn');
    msg.textContent = text;
    overlay.style.display = 'flex';
    const cleanup = () => { overlay.style.display = 'none'; };
    yesBtn.onclick = () => { cleanup(); resolve(true); };
    noBtn.onclick = () => { cleanup(); resolve(false); };
  });
}

// ---------- Resources ----------
async function readResource() {
  const uri = document.getElementById('resourceCustom').value.trim() || document.getElementById('resourceUri').value;
  const r = await fetch('/api/resource/' + encodeURIComponent(uri));
  const data = await r.json();
  showResult(document.getElementById('resourceResult'), data);
}

async function getPrompt() {
  const name = document.getElementById('promptName').value;
  const r = await fetch('/api/prompt/' + encodeURIComponent(name));
  const data = await r.json();
  showResult(document.getElementById('promptResult'), data);
}

// ---------- Settings ----------
async function setTrainingMode() {
  const mode = document.getElementById('trainingMode').value === 'true';
  const r = await mcpCall('set_training_mode', { training: mode });
  showResult(document.getElementById('settingsResult'), r);
}

// ---------- Presets ----------
async function createClassifier() {
  const results = [];
  const r1 = await mcpCall('create_model', { input_shape: [1, 5] });
  results.push(t('preset.create') + ': ' + JSON.stringify(r1));
  if (!r1.error) {
    const r2 = await mcpCall('add_layer', { layer_type: 'dense', input_dim: 5, output_dim: 10 });
    results.push(t('preset.dense') + '(5\u219210): ' + JSON.stringify(r2));
    const r3 = await mcpCall('add_layer', { layer_type: 'relu' });
    results.push('ReLU: ' + JSON.stringify(r3));
    const r4 = await mcpCall('add_layer', { layer_type: 'dense', input_dim: 10, output_dim: 3 });
    results.push(t('preset.dense') + '(10\u21923): ' + JSON.stringify(r4));
    const r5 = await mcpCall('add_layer', { layer_type: 'softmax' });
    results.push('Softmax: ' + JSON.stringify(r5));
  }
  const el = document.getElementById('presetResult');
  el.className = 'result mt-1 success';
  el.textContent = results.join('\n');
  updateModelStatus();
  refreshModelInfo();
}

// ---------- Init ----------
applyLang();
setTimeout(updateModelStatus, 500);
