const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

document.getElementById('clear-btn').onclick = () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

let session;
ort.InferenceSession.create('/Users/islamnashentaev/Desktop/projectX/site_pages/digit_model.onnx').then(s => {
  session = s;
  document.getElementById('result').innerText = 'Модель загружена';
}).catch(err => {
  console.error(err);
  document.getElementById('result').innerText = 'Ошибка загрузки модели';
});

document.getElementById('predict-btn').onclick = async () => {
  if (!session) return;
  // взять пиксели, привести к [1,1,28,28]
  const imageData = ctx.getImageData(0, 0, 280, 280);
  // ресайз до 28×28
  const off = document.createElement('canvas');
  off.width = 28;
  off.height = 28;
  const octx = off.getContext('2d');
  octx.drawImage(canvas, 0, 0, 28, 28);
  const img = octx.getImageData(0, 0, 28, 28).data;
  // подготовить Float32Array
  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28*28; i++) {
    // взять только один канал и нормализовать
    input[i] = (255 - img[i*4]) / 255;
  }
  const tensor = new ort.Tensor('float32', input, [1,1,28,28]);
  const feeds = { input: tensor };
  const results = await session.run(feeds);
  const output = results.output.data;
  const pred = output.indexOf(Math.max(...output));
  document.getElementById('result').innerText = `Результат: ${pred}`;
};
