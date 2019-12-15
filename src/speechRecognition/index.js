let recognizer;

const NUM_FRAMES = 43;
let examples = [];


function collect(label) {
  if (recognizer.isListening()) {
   return recognizer.stopListening();
  }
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   console.log(frameSize);
   let vals = normalize(data.subarray(-frameSize*NUM_FRAMES));
   examples.push({vals, label});
   document.querySelector('#examples').textContent =
       `${examples.length} examples`;
  }, {
   overlapFactor: 0.01,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
  });
}
  
function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}
  
  
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
  
async function train() {
   let startTime = new Date();
   toggleButtons(false);
   const ys = tf.oneHot(examples.map(e => e.label), 3);
   const xsShape = [examples.length, ...INPUT_SHAPE];
   const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
  
   await model.fit(xs, ys, {
     batchSize: 25,
     epochs: 20,
     callbacks: {
       onEpochEnd: (epoch, logs) => {
         document.querySelector('#model-accuracy').textContent =
             `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
       }
     }
   });
   tf.dispose([xs, ys]);
   toggleButtons(true);
   document.getElementById('time').textContent = `Time taken to train model: ${new Date() - startTime} ms`;
}
  
function buildModel() {
   
   model = tf.sequential();
   model.add(tf.layers.depthwiseConv2d({
     depthMultiplier: 8,
     kernelSize: [NUM_FRAMES, 3],
     activation: 'relu',
     inputShape: INPUT_SHAPE
   }));
   model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
   model.add(tf.layers.flatten());
   model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
   const optimizer = tf.train.adam(0.01);
   model.compile({
     optimizer,
     loss: 'categoricalCrossentropy',
     metrics: ['accuracy']
   });
}
  
function toggleButtons(enable) {
   document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}
  
function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}
  
function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listenButton').textContent = 'Listen';
    return;
  }
  toggleButtons(false);
  document.getElementById('listenButton').textContent = 'Stop';
  document.getElementById('listenButton').disabled = false;
   
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    let startTime = new Date();
    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    const probs = model.predict(input);
    const predLabel = probs.argMax(1);
    const label = (await predLabel.data())[0];
    actionHandler(label);
    tf.dispose([input, probs, predLabel]);
    document.getElementById('time').textContent = `Time taken for inference: ${new Date() - startTime} ms`;
  }, {
    overlapFactor: 0.01,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}
  
function actionHandler(label){
  console.log(label);
  if(label == 2){
    document.getElementById('inference').textContent = 'Noise';
    return;
  }
  if(label == 1){
    loadFromReddit('tensorflow');
    document.getElementById('inference').textContent = 'Loading from r/tensorflow';
    return;
  }
  if(label == 0){
    loadFromReddit('programming');
    document.getElementById('inference').textContent = 'Loading from r/programming';
    return;
  }
}

loadFromReddit = (subreddit) => {
    axios.get(`https://www.reddit.com/r/${subreddit}/new.json?limit=10`).then( json => {
        let posts = json.data.data.children;
        let render = '';
        posts.forEach((post) => {
            render += `<div class='post'><h2 class='reddit-title'>${post.data.title}</h2>
            <span class='reddit-author'>u/${post.data.author}</span>
            <span class='reddit-subreddit'>r/${post.data.subreddit}</span>
            <p class='reddit-content'>${post.data.selftext.slice(0, 100)}...</p>
            <a class='reddit-url' href=${post.data.url}>full</a>
            </div>
            `
        });
        document.getElementById('posts').innerHTML = render;
    });
};

changeBackend = () => {
  if(tf.getBackend() == 'cpu'){
    tf.setBackend('webgl');
    document.getElementById('backendButton').textContent = 'Switch to CPU';
  }else{
    tf.setBackend('cpu');
    document.getElementById('backendButton').textContent = 'Switch to WebGL';
  }
};

trainOffload = () => {
  if(examples.length == 0){
    document.querySelector('#model-accuracy').textContent = `Nothing to train on`;
    return;
  }
  toggleButtons(false);
  let startTime = new Date();
  axios.post(`https://10.64.151.156:8000/train`, {examples: JSON.stringify(examples)})
  .then( response => {
    document.querySelector('#model-accuracy').textContent = `Accuracy: ${response.data.accuracy}% after 10 epochs`;
    document.getElementById('time').textContent = `Time taken to train model: ${new Date() - startTime} ms`;
    toggleButtons(true);
  })
  .catch( error => {
    console.log(error);
    toggleButtons(true);
  });
};

listenOffload = () => {
  if (examples.length == 0){
    document.querySelector('#inference').textContent = `Train model first`;
    return;
  }
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listenOffloadButton').textContent = 'Inference server-side';
    return;
  }
  toggleButtons(false);
  document.getElementById('listenOffloadButton').textContent = 'Stop';
  document.getElementById('listenOffloadButton').disabled = false;
   
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    let startTime = new Date();
    axios.post(`https://10.64.151.156:8000/inference`, {spectrogram: JSON.stringify({frameSize, data})})
    .then( response => {
        actionHandler(response.data.result);
        document.getElementById('time').textContent = `Time taken for inference: ${new Date() - startTime} ms`;
    })
    .catch( error => {
      console.log(error);
      toggleButtons(true);
    });
    
  }, {
    overlapFactor: 0.01,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
};


async function app() {
  toggleButtons(false);
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  buildModel();
  toggleButtons(true);
  if(tf.getBackend() == 'cpu'){
    document.getElementById('backendButton').textContent = 'Switch to WebGL';
  }else{
    document.getElementById('backendButton').textContent = 'Switch to CPU';
  }
}
  
app();