import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';
import { MNISTDataLoader } from './mnist-data-loader.ts';

const pass = (message: string): void => {
  console.log(`✅ ${message}`);
};

const fail = (message: string): void => {
  console.log(`❌ ${message}`);
};

const warn = (message: string): void => {
  console.log(`⚠️ ${message}`);
};

const checkFileExists = (filename: string): void => {
  try {
    Deno.lstatSync(filename);
    pass(`${filename} exists.`);
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) {
      fail(`${filename} not found!`);
    } else {
      warn(`Could not check for existence of ${filename}...`);
    }
  }
};

const checkMatrixDimensions = (
  matrix: Matrix,
  name: string,
  rows: number,
  cols: number
): void => {
  const [rowSize, colSize] = matrix.shape;
  if (rowSize === rows && colSize === cols) {
    pass(`Matrix ${name} has dimensions ${rows} X ${cols}.`);
  } else {
    fail(
      `Matrix ${name} has dimensions ${rowSize} X ${colSize}, but expected ${rows} X ${cols}!`
    );
  }
};

const loader = new MNISTDataLoader();

console.log('------------------------');
console.log('Loading training data...');
console.log('------------------------');
checkFileExists('./data/train-images-idx3-ubyte');
checkFileExists('./data/train-labels-idx1-ubyte');
const [X_train, y_train] = await loader.load_train();
checkMatrixDimensions(X_train, 'X_train', 60000, 784);
checkMatrixDimensions(y_train, 'y_train', 60000, 1);
console.log('------------------------\n\n');

console.log('------------------------');
console.log('Loading test data...');
console.log('------------------------');
checkFileExists('./data/t10k-images-idx3-ubyte');
checkFileExists('./data/t10k-labels-idx1-ubyte');
const [X_test, y_test] = await loader.load_test();
checkMatrixDimensions(X_test, 'X_test', 10000, 784);
checkMatrixDimensions(y_test, 'y_test', 10000, 1);
console.log('------------------------\n\n');
