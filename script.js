const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Инициализация холста
function clearCanvas() {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
clearCanvas();

ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

// Обработчики рисования
canvas.addEventListener('mousedown', () => {
  drawing = true;
  ctx.beginPath();
});
canvas.addEventListener('mouseup', () => {
  drawing = false;
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
});

// Кнопка «Очистить»
document.getElementById('clear-btn').onclick = () => {
  clearCanvas();
  document.getElementById('result').innerText = 'Холст очищен. Нарисуйте цифру.';
};

// Пути для загрузки модели
const MODEL_PATHS = [
  './digit_model.onnx',
  'https://cdn.jsdelivr.net/gh/Islam4562/cnnweb.io@main/digit_model.onnx',
  'https://raw.githack.com/Islam4562/cnnweb.io/main/digit_model.onnx'
];

let session = null;

// Загрузка модели
(async () => {
  const resultEl = document.getElementById('result');
  for (const url of MODEL_PATHS) {
    try {
      session = await ort.InferenceSession.create(url);
      resultEl.innerText = `Модель загружена`;
      console.log('Модель успешно загружена из:', url);
      break;
    } catch (e) {
      console.warn(`Ошибка загрузки модели из ${url}:`, e);
    }
  }
  if (!session) {
    resultEl.innerText = 'Ошибка загрузки модели';
  }
})();

// Кнопка «Распознать»
document.getElementById('predict-btn').onclick = async () => {
  const resultEl = document.getElementById('result');
  if (!session) {
    resultEl.innerText = 'Модель не загружена';
    return;
  }

  try {
    // Ресайз холста до 28×28
    const off = document.createElement('canvas');
    off.width = 28;
    off.height = 28;
    const octx = off.getContext('2d');
    octx.drawImage(canvas, 0, 0, 28, 28);
    const img = octx.getImageData(0, 0, 28, 28).data;

    // Подготовка входного тензора
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      input[i] = (255 - img[i * 4]) / 255; // инверсия: белое = 0, чёрное = 1
    }
    const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);

    // Запуск инференса
    const outputMap = await session.run({ input: tensor });
    const output = outputMap.output.data;
    const prediction = output.indexOf(Math.max(...output));

    resultEl.innerText = `Цифра: ${prediction}`;
    setTimeout(() => {
      clearCanvas(); // автоматически очищает холст
      resultEl.innerText = 'Нарисуйте следующую цифру';
    }, 2000); // через 2 секунды всё сбрасывается

  } catch (e) {
    console.error(e);
    resultEl.innerText = 'Ошибка распознавания';
  }
};
