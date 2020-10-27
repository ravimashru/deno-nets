import { MNISTDataLoader } from './mnist-data-loader.ts';
import { Network } from './network.ts';
import { onehotencoder } from './utility.ts';

const runMNISTExample = async () => {
​    const net = new Network([784, 30, 30, 10]);
​    const dataLoader = new MNISTDataLoader();
​    const mnistDataset = await dataLoader.load_train();
​    const X_train = mnistDataset[0]
​    const y_train = mnistDataset[1]
​    const y_train_encoded = onehotencoder(y_train)
​    net.train(X_train, y_train_encoded, 2, 0.01, true);
}

runMNISTExample()
