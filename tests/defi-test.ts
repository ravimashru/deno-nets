import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';
import { assertEquals } from 'https://deno.land/std@0.76.0/testing/asserts.ts';
import { accuracy } from '../mod.ts';
import { defiTestData } from './defi-test-data.ts'

import { Network } from '../src/network.ts';
import { relu, reluPrime } from '../src/utility.ts';


for (const entry of defiTestData) {
    console.log(entry.timeStamp)
}

// each row is a datapoint / each value ...
const featuresTrainingData = new Matrix(
    [
        [1],
        [840],
        [1680],
    ]
);


const labelsTrainingData = new Matrix(
    [
        [32883],
        [32837],
        [32811],
    ]
);


const numberOfInputLayers = 1
const numberOfOutputLayers = 1
const numberOfNeuronsInHiddenLayer1 = 100 // automatically play around with this
const numberOfNeuronsInHiddenLayer2 = 80 // automatically play around with this
// playing around automatically with hyperparameters... // take e.g. 80 percent of so far data as training data... 
// deno 

const epochs = 75 // stop after 1000 epochs to save cpu - this shall be automated in the context with the 80 / 20 HyperParamter optimization issue
const learningRate = 1// depends on scale of features

const net = new Network([numberOfInputLayers, numberOfNeuronsInHiddenLayer1, numberOfNeuronsInHiddenLayer2, numberOfOutputLayers], relu, reluPrime);

await net.train(featuresTrainingData, labelsTrainingData, epochs, learningRate)


const featuresTestData = [2480]

const predictionResult = net.feedforward(featuresTestData)

console.log(predictionResult)

// time = feature 
// prediction = price
// timeseries forecasting vs. regression

// const accuracyScore = accuracy(actual, predicted);

assertEquals(defiTestData.length, 3);
