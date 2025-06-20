const CDN_PATHS = [
  './model.onnx',  // локальный файл рядом с index.html
  'https://cdn.jsdelivr.net/gh/Islam4562/hdrweb.io@main/digit_model.onnx',
  'https://raw.githack.com/Islam4562/hdrweb.io/main/digit_model.onnx'
];

let session = null;
(async () => {
  const resultEl = document.getElementById('result');
  for (const url of CDN_PATHS) {
    try {
      session = await ort.InferenceSession.create(url);
      resultEl.innerText = `Модель загружена из: ${url}`;
      console.log('Loaded ONNX from', url);
      break;
    } catch (e) {
      console.warn(`Не удалось загрузить модель из ${url}:`, e);
    }
  }
  if (!session) {
    resultEl.innerText = 'Ошибка загрузки модели: все варианты упали';
  }
})();
