import { GzipStream } from 'https://deno.land/x/compress@v0.3.3/mod.ts';
import { MNISTDataLoader } from './mnist-data-loader.ts';
import { Network } from './network.ts';
import { onehotencoder } from './utility.ts';

const runMNISTExample = async () => {
    const net = new Network([784, 30, 10]);
    const dataLoader = new MNISTDataLoader();
    const mnistDataset = await dataLoader.load_train();
    const X_train = mnistDataset[0]
    const y_train = mnistDataset[1]
    const y_train_encoded = onehotencoder(y_train)
    net.train(X_train, y_train_encoded, 100001, 1);
}

runMNISTExample()
