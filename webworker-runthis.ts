import { Matrix } from 'https://deno.land/x/math@v1.1.0/mod.ts';

const worker = new Worker(new URL("worker.ts", import.meta.url).href, { type: "module", deno: true });

let mat1 = new Matrix([
    [1, 2],
    [3, 4]
  ]);

  let mat2 = new Matrix([
    [1, 1],
    [1, 1]
  ]);

  console.log(mat1.times(mat2));

  worker.postMessage({ mat1:mat1 , mat2:mat2 });

  worker.onmessage = function(e){
    console.log("Working!!!");
    console.log(new Matrix(e.data.matrix));
  }
