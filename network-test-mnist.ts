import { GzipStream } from 'https://deno.land/x/compress@v0.3.3/mod.ts';
import { MNISTDataLoader } from './mnist-data-loader.ts';
import { Network } from './network.ts';

const runMNISTExample = async () => {
    const net = new Network([784, 30, 10]);
    const dataLoader = new MNISTDataLoader();
    const mnistDataset = await dataLoader.load_train();
    const X_train = mnistDataset[0]
    const y_train = mnistDataset[1]
    net.train(X_train, y_train, 100001, 1);
}

runMNISTExample()
