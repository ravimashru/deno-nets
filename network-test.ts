import { Matrix } from 'https://deno.land/x/math/mod.ts';
import { Network } from './network.ts';

const net = new Network([2, 5, 2]);

const X_train = new Matrix([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);

const y_train = new Matrix([
  [0, 0],
  [0, 1],
  [0, 1],
  [1, 1]
]);

net.train(X_train, y_train, 100000, 1, true)
