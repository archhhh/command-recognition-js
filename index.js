const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const fs = require('fs');
const https = require('https');
const privateKey  = fs.readFileSync('./key.pem');
const certificate = fs.readFileSync('./cert.pem');
const credentials = {key: privateKey, cert: certificate};

const app = express();
const port = 8000;
const bodyParser = require('body-parser');

const NUM_FRAMES = 43;

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
  
async function train(examples) {
   const ys = tf.oneHot(examples.map(e => e.label), 3);
   const xsShape = [examples.length, ...INPUT_SHAPE];
   const xs = tf.tensor(flatten(examples.map(e => {
      let vals = [];
      Object.keys(e.vals).forEach( key => vals.push(e.vals[key]));
      return vals;
   })), xsShape);
   let accuracy;
   await model.fit(xs, ys, {
     batchSize: 25,
     epochs: 20,
     callbacks: {
       onEpochEnd: (epoch, logs) => {
            accuracy = (logs.acc * 100).toFixed(1);
            console.log(`Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`);
       }
     }
   });
   await model.save('file://./src/model');
   tf.dispose([xs, ys]);
   return accuracy;
}

function normalize(x) {
    const mean = -100;
    const std = 10;
    return x.map(x => (x - mean) / std);
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
  

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}


async function inference(spectrogram){
    let data = [];
    Object.keys(spectrogram.data).forEach(key => data.push(spectrogram.data[key]));
    const vals = normalize(data.slice(-spectrogram.frameSize * NUM_FRAMES));
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    const probs = model.predict(input);
    const predLabel = probs.argMax(1);
    tf.dispose([input, probs]);
    return (await predLabel.data())[0];
}


buildModel();

app.use(express.static('src'));
app.use(bodyParser.json({limit: '200MB', extended: true}));
app.use(bodyParser.urlencoded({limit: '200MB', extended: true}));

app.post('/train', (req, res) => {
    let startTime = new Date();
    train(JSON.parse(req.body.examples))
    .then( accuracy => res.json({accuracy: accuracy, time: new Date() - startTime}))
    .catch( error => { console.log(error); res.status(400).send(error); });
});
app.post('/inference', async (req, res) => {
    let startTime = new Date();
    res.json({result: await inference(JSON.parse(req.body.spectrogram)), time: new Date() - startTime});
});

const httpsServer = https.createServer(credentials, app);

httpsServer.listen(port, () => console.log(`Example app listening on port ${port}!`))
