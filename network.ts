/**
 * Reference: http://neuralnetworksanddeeplearning.com
 */

import { Matrix } from 'https://deno.land/x/math/mod.ts';
import {
  createRandomRealMatrix,
  createZerosMatrix,
  sigmoid,
  sigmoidPrime,
  operateOnMatrix,
  addMatrixArrays,
  shuffle,
  createMiniBatches,
  printResults,
} from './utility.ts';

export class Network {
  private layer_sizes: number[] = [];
  private biases: Matrix[] = [];
  private weights: Matrix[] = [];

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
      const weightMatrix = this.weights[i];
      const biasMatrix = this.biases[i];
      res = weightMatrix.times(res).plus(biasMatrix);
      const [rowSize, colSize] = res.shape;

      for (let j = 0; j < rowSize; j++) {
        for (let k = 0; k < colSize; k++) {
          res.matrix[j][k] = sigmoid(res.matrix[j][k]);
        }
      }
    }

    return res.transpose().matrix;
  }

  public train(X_train: Matrix, y_train: Matrix, epochs: number, lr: number) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle X_train, y_train after every epoch
      const ArrayX_Y = shuffle(X_train, y_train);
      let miniBatchesX = createMiniBatches(ArrayX_Y[0], 10);
      let miniBatchesY = createMiniBatches(ArrayX_Y[1], 10);
      
      for (let index = 0; index < miniBatchesX.length; index++) {
        const miniBatchX = miniBatchesX[index];
        const miniBatchY = miniBatchesY[index]
        this.update_mini_batch(miniBatchX, miniBatchY, lr)
      }
      if (epoch % 25000 === 0) {
        console.log(`Epoch ${epoch}:`)
        printResults(X_train, this)
        console.log()
      }
      
    }
  }

  public backprop(x: Matrix, y: Matrix) {
    if (x.shape[1] !== this.layer_sizes[0]) {
      throw new Error(
        `Input should have ${this.layer_sizes[0]} features, ${x.shape[1]} features passed!`
      );
    }
    if (y.shape[1] !== this.layer_sizes[this.layer_sizes.length - 1]) {
      throw new Error(
        `Output should have ${
          this.layer_sizes[this.layer_sizes.length - 1]
        } labels, ${y.shape[1]} features passed!`
      );
    }

    const grad_w: Matrix[] = [];
    const grad_b: Matrix[] = [];

    let activation = x;
    let z = new Matrix([[]]);
    const activations = [x];
    const zVectors = [];

    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];

      // console.log('------');
      // console.log(activation);
      // console.log(weights);
      // console.log(biases);

      z = weights.times(activation.transpose()).plus(biases);
      zVectors.push(z);
      activation = operateOnMatrix(z, sigmoid).transpose();
      activations.push(activation);
    }

    // console.log('<<<<>>>>><<<<<>>>><<<>>>');
    // console.log('activations', activations);
    // console.log('y', y);
    // console.log('z', zVectors);

    // (1x2) - (1x2) [dot] (2 X 1)
    // Need: 1 x 2

    // console.log(activation);
    // console.log();
    // console.log(y);
    // console.log();
    // console.log(activation.minus(y));
    // console.log();
    // console.log(activation.minus(y).transpose());
    // console.log();
    // console.log(activation.minus(y).transpose().times(z));
    // console.log();
    // console.log(z);

    let db = activation
      .minus(y)
      .transpose()
      .times(operateOnMatrix(z, sigmoidPrime));
    // console.log('partial derivative', db);

    grad_b.unshift(db);

    // console.log('z shapes');
    // printShapes(zVectors);

    // console.log('Activation shapes');
    // printShapes(activations);

    // console.log('Weight shapes');
    // printShapes(this.weights);

    // console.log('Bias shapes');
    // printShapes(this.biases);

    const partialDerivativeWeights = db.times(
      activations[activations.length - 2]
    );

    grad_w.unshift(partialDerivativeWeights);

    for (let i = this.weights.length - 2; i >= 0; i--) {
      z = zVectors[i];
      const zSigmoidPrime = operateOnMatrix(z, sigmoidPrime);

      db = this.weights[i + 1].transpose().times(db).times(zSigmoidPrime);
      grad_b.unshift(db);

      const dw = db.times(activations[i]);
      grad_w.unshift(dw);
    }

    // console.log('grad w shapes');
    // printShapes(grad_w);

    // console.log('grad b shapes');
    // printShapes(grad_b);

    return [grad_w, grad_b];
  }

  public update_mini_batch(X: Matrix, y: Matrix, lr: number): void {
    let weightUpdates: Matrix[] = [];
    let biasUpdates: Matrix[] = [];

    for (let i = 0; i < this.weights.length; i++) {
      weightUpdates.push(createZerosMatrix(...this.weights[i].shape));
      biasUpdates.push(createZerosMatrix(...this.biases[i].shape));
    }

    for (let i = 0; i < X.matrix.length; i++) {
      const inputs = X.row(i);
      const outputs = y.row(i);

      const [dw, db] = this.backprop(
        new Matrix([inputs]),
        new Matrix([outputs])
      );

      weightUpdates = addMatrixArrays(weightUpdates, dw);
      biasUpdates = addMatrixArrays(biasUpdates, db);
    }

    // Update weights
    for (let i = 0; i < this.weights.length; i++) {
      const currentWeights = this.weights[i];
      const deltaWeights = weightUpdates[i];

      this.weights[i] = currentWeights.minus(
        deltaWeights.times(lr / X.matrix.length)
      );
    }

    // Update biases
    for (let i = 0; i < this.biases.length; i++) {
      const currentBiases = this.biases[i];
      const deltaBiases = biasUpdates[i];

      this.biases[i] = currentBiases.minus(
        deltaBiases.times(lr / X.matrix.length)
      );
    }
  }
}
