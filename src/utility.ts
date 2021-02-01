import { Matrix } from 'https://deno.land/x/math@v1.1.0/mod.ts';
// import { Network } from 'https://deno.land/x/deno_nets/network.ts';
import { Network } from './network.ts';

export const createRandomRealMatrix = (dim1: number, dim2: number): Matrix => {
  const randomNumbers = [];

  for (let i = 0; i < dim1 * dim2; i++) {
    randomNumbers.push(Math.random());
  }

  const sum = randomNumbers.reduce((a, b) => a + b, 0);
  const avg = sum / randomNumbers.length;
  const variance =
    randomNumbers
      .map((num) => num - avg) // Difference from mean
      .map((num) => num * num) // Squared
      .reduce((a, b) => a + b, 0) / randomNumbers.length; // Average

  const stddev = Math.sqrt(variance);

  const normalizedRandomNumbers = randomNumbers.map(num => (num - avg) / stddev);

  return new Matrix([normalizedRandomNumbers]).reshape([dim1, dim2]);
};

export const createZerosMatrix = (dim1: number, dim2: number): Matrix => {
  const res: number[][] = [];
  for (let i = 0; i < dim1; i++) {
    const inner = [];
    for (let j = 0; j < dim2; j++) {
      inner.push(0);
    }
    res.push(inner);
  }
  return new Matrix(res);
};

export const sigmoid = (value: number): number => {
  return 1.0 / (1.0 + Math.exp(-1 * value));
};

export const sigmoidPrime = (value: number): number => {
  return sigmoid(value) * (1 - sigmoid(value));
};

export const tanh = (value: number): number => {
  return Math.tanh(value);
};

export const tanhPrime = (value: number): number => {
  return 1 - (Math.tanh(value) * Math.tanh(value));
};

export const relu = (value: number): number => {
  return value > 0 ? value : 0.00002;
}

export const reluPrime = (value: number): number => {
  // return value > 0 ? value : 0;
  return (value > 0) ? 1 : 0.00002;
}

export const operateOnMatrix = (matrix: Matrix, fn: Function): Matrix => {
  const [rows, cols] = matrix.shape;
  const res = [];

  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(fn(matrix.pointAt(i, j)));
    }
    res.push(row);
  }

  return new Matrix(res);
};

export const printShapes = (arr: Matrix[]): void => {
  arr.forEach((mat: Matrix) => {
    const [rows, cols] = mat.shape;
    console.log(`${rows} X ${cols}`);
  });
};

export const addMatrixArrays = (arr1: Matrix[], arr2: Matrix[]): Matrix[] => {
  if (arr1.length !== arr2.length) {
    throw new Error(
      `Arrays have different lengths: ${arr1.length} and ${arr2.length}!`
    );
  }

  const res: Matrix[] = [];

  for (let i = 0; i < arr1.length; i++) {
    res.push(arr1[i].plus(arr2[i]));
  }

  return res;
};

export const shuffle = (matrix1: Matrix, matrix2: Matrix): Matrix[] => {
  const [rows, cols] = matrix1.shape;
  const res: Matrix[] = [];
  let randIndex: number, temp1: any, temp2: any;
  let index = rows;

  while (index) {
    randIndex = Math.floor(Math.random() * index);
    index -= 1;
    temp1 = matrix1.matrix[index]
    temp2 = matrix2.matrix[index]
    matrix1.matrix[index] = matrix1.matrix[randIndex]
    matrix2.matrix[index] = matrix2.matrix[randIndex]
    matrix1.matrix[randIndex] = temp1;
    matrix2.matrix[randIndex] = temp2;
  }

  res.push(matrix1, matrix2)
  return res;
}

export const createMiniBatches = (matrix: Matrix, chunk: number): Matrix[] => {
  const [rows, cols] = matrix.shape;
  const res = [];
  let temp = [];

  for (let i = 0; i < rows; i += chunk) {
    temp = matrix.valueOf().slice(i, i + chunk)
    res.push(new Matrix(temp));
  }
  return res;
}

export const printResults = (X_train: Matrix, net: Network) => {
  for (let i = 0; i < X_train.matrix.length; i++) {
    const input = X_train.row(i);
    const result = net.feedforward(input);
    console.log(`Inputs: ${input}\tOutput: ${result}`);
  }
}

export const onehotencoder = (labels: Matrix): Matrix => {
  if (labels.shape[1] !== 1) {
    throw new Error('Only single label columns can be encoded!');
  }

  // Assumption: labels matrix has only one column
  const labelsArray = labels.transpose().row(0);
  const minLabel = Math.min(...labelsArray);
  const maxLabel = Math.max(...labelsArray);
  
  const oheMatrix = createZerosMatrix(labels.shape[0], maxLabel - minLabel + 1);
  for (let i = 0; i < labels.shape[0]; i++) {
    const label = labelsArray[i];
    oheMatrix.matrix[i][label - minLabel] = 1;
  }
  return oheMatrix;
}
