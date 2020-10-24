/**
 * Reference: http://neuralnetworksanddeeplearning.com
 */

import { Matrix } from 'https://deno.land/x/math/mod.ts';

const createRandomRealMatrix = (dim1: number, dim2: number) => {
  const res: number[][] = [];
  for (let i = 0; i < dim1; i++) {
    const inner = [];
    for (let j = 0; j < dim2; j++) {
      inner.push(Math.random());
    }
    res.push(inner);
  }
  return res;
};

const sigmoid = (value: number) => {
  return 1.0 / (1.0 + Math.exp(-1 * value));
};

class Network {
  private layer_sizes: number[] = [];
  private biases: number[][][] = [];
  private weights: number[][][] = [];

  constructor(layer_sizes: number[]) {
    if (layer_sizes.length < 3) {
      throw new Error(`Please create a network with at least 3 layers!`);
    }

    this.layer_sizes = layer_sizes;

    // Create weight and bias matrices
    for (let i = 1; i < layer_sizes.length; i++) {
      const dim1 = layer_sizes[i];
      const dim2 = layer_sizes[i - 1];

      this.weights[i - 1] = createRandomRealMatrix(dim1, dim2);
      this.biases[i - 1] = createRandomRealMatrix(dim1, 1);
    }
  }

  /**
   * Return the output of the network when `inputs` are fed
   */
  public feedforward(inputs: number[]) {
    // Ensure input matches input layer size
    if (inputs.length !== this.layer_sizes[0]) {
      throw new Error(
        `Dimension of input not equal to value expected by network!`
      );
    }

    // Forward propagation
    let res: any = new Matrix([inputs]).transpose();
    for (let i = 0; i < this.weights.length; i++) {
      const weightMatrix = new Matrix(this.weights[i]);
      const biasMatrix = new Matrix(this.biases[i]);
      res = weightMatrix.times(res).plus(biasMatrix);
      const [rowSize, colSize] = res.shape;

      for (let j = 0; j < rowSize; j++) {
        for (let k = 0; k < colSize; k++) {
          res.matrix[j][k] = sigmoid(res.matrix[j][k]);
        }
      }
    }

    return res.transpose().matrix[0];
  }

  public train(X: Matrix, y: Matrix, ) {}

  private backprop(x: number[], y: number[]) {}

}

const net = new Network([2, 3, 1]);

console.log(net.feedforward([2, 3]));
