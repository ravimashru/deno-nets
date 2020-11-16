import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';

export const accuracy = (actual: Matrix, predicted: Matrix) => {

  const total = actual.shape[0];
  let equal = 0;

  for (let i = 0; i < actual.shape[0]; i++) {
    const actualLabel = actual.row(i).reduce((iMax, e, i, arr) => e > arr[iMax] ? i: iMax, 0);
    const predictedLabel = predicted.row(i).reduce((iMax, e, i, arr) => e > arr[iMax] ? i: iMax, 0);
    if (actualLabel === predictedLabel) {
      equal++;
    }
  }
  return equal / total;
};
