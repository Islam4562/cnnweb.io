const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Инициализация холста
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

// Обработчики рисования
canvas.addEventListener('mousedown', () => {
  drawing = true;
  ctx.beginPath();
});
canvas.addEventListener('mouseup', () => { drawing = false; });
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
});

// Кнопка «Очистить»
document.getElementById('clear-btn').onclick = () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

// Пути для загрузки модели
const MODEL_PATHS = [
  './digit_model.onnx',
  'https://cdn.jsdelivr.net/gh/Islam4562/hdrweb.io@main/digit_model.onnx',
  'https://raw.githack.com/Islam4562/hdrweb.io/main/digit_model.onnx'
];

let session = null;
(async () => {
  const resultEl = document.getElementById('result');
  for (const url of MODEL_PATHS) {
    try {
      session = await ort.InferenceSession.create(url);
      resultEl.innerText = `Модель загружена из: ${url}`;
      console.log('Загружено ONNX из', url);
      break;
    } catch (e) {
      console.warn(`Не удалось загрузить модель из ${url}:`, e);
    }
  }
  if (!session) {
    resultEl.innerText = 'Ошибка загрузки модели: все варианты упали';
  }
})();

// Кнопка «Распознать»
document.getElementById('predict-btn').onclick = async () => {
  const resultEl = document.getElementById('result');
  if (!session) return;
  try {
    // Ресайз холста до 28×28
    const off = document.createElement('canvas');
    off.width = 28; off.height = 28;
    const octx = off.getContext('2d');
    octx.drawImage(canvas, 0, 0, 28, 28);
    const img = octx.getImageData(0, 0, 28, 28).data;

    // Формирование входного тензора [1,1,28,28]
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      input[i] = (255 - img[i * 4]) / 255;
    }
    const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);

    // Инференс
    const outputMap = await session.run({ input: tensor });
    const output = outputMap.output.data;
    const pred = output.indexOf(Math.max(...output));
    resultEl.innerText = `Результат: ${pred}`;
  } catch (e) {
    console.error(e);
    resultEl.innerText = 'Ошибка инференса';
  }
};
