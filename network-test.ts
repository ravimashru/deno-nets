import { Matrix } from 'https://deno.land/x/math@v1.1.0/mod.ts';
import { Network } from './network.ts';
import { shuffle, createMiniBatches } from './utility.ts';

const net = new Network([2, 5, 2]);

const X_train = new Matrix([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
// const X_train = new Matrix([
//     [50, 1],
//     [3, 2],
//     [4, 3],
//     [15, 4],
//     [2, 5],
//     [33, 6],
//     [42, 7],
//   ]);
//   const y_train = new Matrix([
//     [50, 1],
//     [3, 2],
//     [4, 3],
//     [15, 4],
//     [2, 5],
//     [33, 6],
//     [42, 7],
//   ]);

const y_train = new Matrix([
  [0, 0],
  [0, 1],
  [0, 1],
  [1, 1]
]);

// console.log(shuffle(X_train, y_train))
// let data = createMiniBatches(X_train, 3)
// console.log(data)


// const printResults = () => {
//   for (let i = 0; i < X_train.matrix.length; i++) {
//     const input = X_train.row(i);
//     const result = net.feedforward(input);
//     console.log(`Inputs: ${input}\tOutput: ${result}`);
//   }
// }

// for (let epoch = 0; epoch <= 100000; epoch++) {
//   net.update_mini_batch(X_train, y_train, 1);
//   if (epoch % 25000 === 0) {
//     console.log(`Epoch ${epoch}:`)
//     printResults();
//     console.log();
//   }
// }

// train(X_train, y_train, epochs, learningRate)
net.train(X_train, y_train, 100000, 1, true);
