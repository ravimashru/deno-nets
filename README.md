# deno-nets
Create, train and use neural networks using Typescript in Deno

Goal: create a Deno module with an interface like Scikit-learn to create, train and use neural networks

Plan: work on this during the upcoming [Deno Hacktoberfest](https://organize.mlh.io/participants/events/5363-nest-land-hacktoberfest-online-meetup-with-ryan-dahl-sam-williams-and-michael-spengler)

## MVP
- [ ] Dense Layers
- [ ] SGD optimizer
- [ ] Ability to define activation functions (in hidden layers and output layer)
- [ ] Basic metrics (e.g. RMSE for regression, accuracy for classification)
- [ ] A network that achieves decent test accuracy on [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)

## Potential Features
- [ ] Convolutional Layers
- [ ] Different optimizers (e.g. AdaGrad, Adam, SGD with momentum, etc.)
- [ ] Advanced metrics (e.g. F1 score)

## Potential Interface
```typescript
const net = new Network(input_dimensions=5, output_dimensions=1, hidden_layers=[5, 6])  

// X has 2 dimensions (batch_size, input_dimensions)
// y as 2 dimensions (batch_size, output_dimensions)
net.train(X, y)   

// X has 2 dimensions (batch_size, input_dimensions)
net.predict(X)
```

## Loading MNIST Handwritten Digits data
This repository contains the [MNIST handritten digits](http://yann.lecun.com/exdb/mnist/) dataset in the `data` directory to train the network on. The dataset is compressed (gzip) and needs to be uncompressed before it can be used.

The `MNISTDataLoader` class can be used as follows to load data in a format that can be used with the neural network directly:

```typescript
// Create an instance of the loader class
const loader = new MNISTDataLoader();

// Load the training data
const [X_train, y_train] = await loader.load_train();

// Load the test data
const [X_test, y_test] = await loader.load_test();
```

To check if everything works, run the data loader tests:
```shell
$ deno run --allow-read deno-loader-test.ts
```


## Reference
- http://neuralnetworksanddeeplearning.com/
