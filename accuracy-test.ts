import { Matrix } from "https://deno.land/x/math@v1.1.0/matrix/matrix.ts";
import { assertEquals } from "https://deno.land/std@0.76.0/testing/asserts.ts";
import { accuracy } from "./metrics.ts";

const actual = new Matrix([
  [1, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 1, 0, 0]
]);

const predicted = new Matrix([
  [0.9, 0, 0.1, 0],
  [0.05, 0.2, 0.05, 0.8],
  [0.1, 0.6, 0.2, 0.1],
  [0, 1, 0, 0]
]);

const accuracyScore = accuracy(actual, predicted);

assertEquals(accuracyScore, 0.75);