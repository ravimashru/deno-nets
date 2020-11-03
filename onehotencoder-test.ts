import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';
import { onehotencoder } from './utility.ts';
import { assertEquals } from 'https://deno.land/std/testing/asserts.ts';

const testMatrix1 = new Matrix([[6], [1], [2], [3], [4], [5]]);

Deno.test({
  name: 'Check dimensions',
  fn(): void {
    const ohe = onehotencoder(testMatrix1);
    assertEquals(ohe.shape[0], 6);
    assertEquals(ohe.shape[1], 6);
  },
});

const checkMatrixEquality = (mat1: Matrix, mat2: Matrix) => {
  assertEquals(mat1.shape[0], mat2.shape[0]);
  assertEquals(mat1.shape[1], mat2.shape[1]);

  for (let i = 0; i < mat1.shape[0]; i++) {
    for (let j = 0; j < mat1.shape[1]; j++) {
      assertEquals(mat1.pointAt(i, j), mat2.pointAt(i, j));
    }
  }
}

Deno.test({
  name: 'Check encoded values',
  fn(): void {
    const ohe = onehotencoder(testMatrix1);
    const expectedMatrix = new Matrix([
      [0, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
    ]);
    try {
      checkMatrixEquality(expectedMatrix, ohe);
    } catch (e) {
      console.log(`\nActual:\n${ohe}`);
      console.log(`Expected:\n${expectedMatrix}`);
      throw e;
    }
  }
});
