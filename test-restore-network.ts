/**
 * Inputs: 0,1	Output: -0.9779159374565091
 * Inputs: 1,1	Output: 0.9582055274367237
 * Inputs: 1,0	Output: -0.9774049102089323
 * Inputs: 0,0	Output: -0.9998650077535737
 */
import { Matrix } from "https://deno.land/x/math@v1.1.0/matrix/matrix.ts";
import { Network } from "./network.ts";
import { printResults } from "./utility.ts";

const network = await Network.restore("and-gate.network");

const X_train = new Matrix([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);

printResults(X_train, network);
