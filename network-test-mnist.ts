import { PNGImage } from 'https://deno.land/x/dpng@0.7.5/lib/PNGImage.ts';
import { MNISTDataLoader } from './mnist-data-loader.ts';
import { Network } from './network.ts';
import { onehotencoder } from './utility.ts';

const runMNISTExample = async () => {
  const net = new Network([784, 10, 10]);
  const dataLoader = new MNISTDataLoader();
  
  console.info('Loading data...');
  const mnistDataset = await dataLoader.load_test();

  const X_train = mnistDataset[0];
  const y_train = mnistDataset[1];
  
  console.info('Encoding labels...');
  const y_train_encoded = onehotencoder(y_train);

  console.info('Training network...');
  net.train(X_train, y_train_encoded, 2, 0.01, true);

  console.info('Making predictions...');
  const testSize = 10;
  for (let i = 0; i < testSize; i++) {
    const png = new PNGImage(28, 28);
    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        const color = X_train.row(i)[28 * y + x];
        const black = png.createRGBColor({
          r: 255 - color,
          g: 255 - color,
          b: 255 - color,
          a: 1,
        });
        png.setPixel(x, y, black);
      }
    }
    Deno.writeFileSync(
      `./images/${i}_${y_train_encoded.row(i).indexOf(1)}.png`,
      png.getBuffer()
    );
    const res: number[] = net.feedforward(X_train.row(i))[0];
    console.log(`Prediction ${i}: ${res}`);
    console.log(`Max index: ${res.reduce((iMax, e, i, arr) => e > arr[iMax] ? i : iMax, 0)}`);
    console.log();
  }

  // Save network
  await net.save('mnist.network');
};

runMNISTExample();
