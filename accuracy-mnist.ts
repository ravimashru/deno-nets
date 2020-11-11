import { Matrix } from "https://deno.land/x/math@v1.1.0/matrix/matrix.ts";
import { accuracy } from "./metrics.ts";
import { MNISTDataLoader } from "./mnist-data-loader.ts";
import { Network } from "./network.ts";
import { onehotencoder } from "./utility.ts";

const dataLoader = new MNISTDataLoader();

console.info('Loading data...');
const mnistDataset = await dataLoader.load_test();

const X_train = mnistDataset[0];
const y_train = mnistDataset[1];

console.info('Encoding labels...');
const y_train_encoded = onehotencoder(y_train);

// const net = new Network([784, 30, 10]);
const net = await Network.restore('sample-mnist-fc.json');

console.info('Check accuracy predictions...');
const predictions = [];
for (let i = 0; i < X_train.shape[0]; i++) {
  const res = net.feedforward(X_train.row(i));
  predictions.push(res[0]);
}



const acc = accuracy(y_train_encoded, new Matrix(predictions));
console.log(acc);

// console.info('Load network...');
// net.train(X_train, y_train_encoded, 1, 0.01, true);