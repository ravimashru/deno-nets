import { GzipStream } from 'https://deno.land/x/compress@v0.3.3/mod.ts';
import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';
import { PNGImage } from 'https://deno.land/x/dpng@0.7.5/mod.ts';

// ------------------------------------
// Example 1: Uncompressing data
// ------------------------------------
const gzip = new GzipStream();

gzip.on('progress', (prog: string) => {
  console.log(prog);
});

const imagesUnzipped = await gzip.uncompress('./data/train-images-idx3-ubyte.gz', './data/train-images-idx3-ubyte');
const labelsUnzipped = await gzip.uncompress('./data/train-labels-idx1-ubyte.gz', './data/train-labels-idx1-ubyte');

const images = await Deno.readFile('./data/train-images-idx3-ubyte');
const labels = await Deno.readFile('./data/train-labels-idx1-ubyte');

// ------------------------------------

// ------------------------------------
// Example 2: Load images and labels
// Reference: http://yann.lecun.com/exdb/mnist/
//            https://stackoverflow.com/questions/25024179/reading-mnist-dataset-with-javascript-node-js
// ------------------------------------

const unsignedInt8ToInt32 = (unsingedInt8Array: Uint8Array) => {
  let res = 0;
  for (let i = 0; i < unsingedInt8Array.length; i++) {
    res = res * 256 + unsingedInt8Array[i];
  }
  return res;
};

const loadImages = async (filename: string) => {
  const images = await Deno.readFile(filename);

  const itemCount = unsignedInt8ToInt32(images.slice(4, 8));
  const rowCount = unsignedInt8ToInt32(images.slice(8, 12));
  const colCount = unsignedInt8ToInt32(images.slice(12, 16));

  var pixelValues = [];

  for (let image = 0; image < itemCount; image++) {
    var pixels = [];

    for (let y = 0; y < colCount; y++) {
      for (let x = 0; x < rowCount; x++) {
        pixels.push(images[image * 28 * 28 + (x + y * 28) + 16]);
      }
    }

    pixelValues.push(pixels);
  }

  return new Matrix(pixelValues);
};

const loadLabels = async (filename: string) => {
  const labels = await Deno.readFile(filename);

  const labelData = Array.from(labels.slice(8));

  return new Matrix([labelData]).transpose();
};

const X_train = await loadImages('./data/train-images-idx3-ubyte');
const y_train = await loadLabels('./data/train-labels-idx1-ubyte');
console.log(X_train.shape);
console.log(y_train.shape);


// ------------------------------------
// Example 3: Visualize data
// ------------------------------------

const testSize = 10;
for (let i = 0; i < testSize; i++) {
  const png = new PNGImage(28, 28);
  for (let y = 0; y <= 27; y++) {
    for (let x = 0; x <= 27; x++) {
      const color = X_train.row(i)[28 * y + x];
      const black = png.createRGBColor({
        r: 255 - color,
        g: 255 - color,
        b: 255 - color,
        a: 1,
      });
      png.setPixel(x, y, black);
    }
  }
  Deno.writeFileSync(`./images/${y_train.row(i)[0]}_${i}.png`, png.getBuffer());
}
