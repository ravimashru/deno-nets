import { Matrix } from 'https://deno.land/x/math/mod.ts';
import { Network, printResults } from '../mod.ts';

const runLogicGateExample = async () => {
  const net = new Network([2, 5, 1]);
  const X_train = new Matrix([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
  ]);
  const y_train = new Matrix([[-1.0], [-1.0], [-1.0], [1.0]]);
  net.train(X_train, y_train, 10001, 0.03);
  printResults(X_train, net);
  await net.save('and-gate.network');
};

runLogicGateExample();
