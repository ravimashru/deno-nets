/**
 * Webworker implementation for parallelization of the neural net
 * backprop method. Parallel processing of matrices will improve performance.
 * Not included in the current implementation as the deno webworker implementation
 * is still not reliable enough.
 * External types are stripped while being passed to the webwork, hence explicit casts are needed.
 */


import { Matrix } from 'https://deno.land/x/math@v1.1.0/mod.ts';
import { Network } from '../mod.ts';

self.onmessage = async (e) => {
    let x :Matrix = new Matrix(e.data.x.matrix);
    let y :Matrix = new Matrix(e.data.y.matrix);
    let layer_sizes = e.data.layer_sizes;
    let weightMatrix:Matrix[] = [];
    let biasMatrix:Matrix[] = [];
    for(let i=0; i<e.data.weightMatrix.length;i++){
        weightMatrix.push(new Matrix(e.data.weightMatrix[i].matrix));
        biasMatrix.push(new Matrix(e.data.biasMatrix[i].matrix));
    }

    const network = new Network([1,2,3]);

    const [dw, db] = network.backprop(x, y, layer_sizes, weightMatrix, biasMatrix);

    let result = {dw:dw, db:db};

    postMessage(result);

};



