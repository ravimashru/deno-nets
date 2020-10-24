import { GzipStream } from "https://deno.land/x/compress@v0.3.3/mod.ts";


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

