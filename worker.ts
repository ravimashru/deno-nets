import { Matrix } from 'https://deno.land/x/math@v1.1.0/mod.ts';

self.onmessage = async (e) => {
    let mat1 :Matrix = new Matrix(e.data.mat1.matrix);
    let mat2 :Matrix = new Matrix(e.data.mat2.matrix);
    console.log(e);
    console.log(mat1);
    console.log(mat2);
    let result = mat1.times(mat2);
    console.log(result);
    postMessage(result);
    self.close();
};
