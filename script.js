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
canvas.addEventListener('mouseup', () => drawing = false);
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

// Загрузка модели через fetch + ONNX Runtime Web
let session = null;
(async () => {
  try {
    const resp = await fetch('digit_model.onnx.onnx');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const buffer = await resp.arrayBuffer();
    session = await ort.InferenceSession.create(buffer);
    document.getElementById('result').innerText = 'Модель загружена';
  } catch (e) {
    console.error(e);
    document.getElementById('result').innerText = 'Ошибка загрузки модели';
  }
})();

// Кнопка «Распознать»
document.getElementById('predict-btn').onclick = async () => {
  if (!session) return;
  // Ресайз холста до 28×28
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  off.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
  const img = off.getContext('2d').getImageData(0, 0, 28, 28).data;
  
  // Формирование входного тензора [1,1,28,28]
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    input[i] = (255 - img[i * 4]) / 255;
  }
  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
  
  // Инференс
  try {
    const outputMap = await session.run({ input: tensor });
    const output = outputMap.output.data;
    const pred = output.indexOf(Math.max(...output));
    document.getElementById('result').innerText = `Результат: ${pred}`;
  } catch (e) {
    console.error(e);
    document.getElementById('result').innerText = 'Ошибка инференса';
  }
};
