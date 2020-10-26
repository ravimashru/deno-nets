import { Matrix } from 'https://deno.land/x/math/mod.ts';
import { MNISTDataLoader } from './mnist-data-loader.ts';
import { Network } from './network.ts';


const runMNISTExample = async () => {
    const net = new Network([2, 5, 3]);
    const dataLoader = new MNISTDataLoader();
    const mnistDataset = await dataLoader.load_train();
    const X_train = mnistDataset[0]
    const y_train = mnistDataset[1]
    net.train(X_train, y_train, 100001, 1);
}

const runLogicGateExample = async () => {
    const net = new Network([2, 5, 3]);
    const X_train = new Matrix([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y_train = new Matrix([
      [-1, -1, -1],
      [-1, 1, 1],
      [-1, 1, 1],
      [1, 1, -1],
    ]);
    net.train(X_train, y_train, 100001, 1);
}

// runMNISTExample();
runLogicGateExample();





