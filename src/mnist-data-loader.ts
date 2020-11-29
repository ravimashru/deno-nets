import { Matrix } from 'https://deno.land/x/math@v1.1.0/matrix/matrix.ts';

export class MNISTDataLoader {
  public async load_train(): Promise<[Matrix, Matrix]> {
    const basePath = (new URL('.', import.meta.url)).pathname;
    const X_train = await this.loadImages(basePath + '../data/train-images-idx3-ubyte');
    const y_train = await this.loadLabels(basePath + '../data/train-labels-idx1-ubyte');
    return [X_train, y_train];
  }
  public async load_test(): Promise<[Matrix, Matrix]> {
    const basePath = (new URL('.', import.meta.url)).pathname;
    const X_test = await this.loadImages(basePath + '../data/t10k-images-idx3-ubyte');
    const y_test = await this.loadLabels(basePath + '../data/t10k-labels-idx1-ubyte');
    return [X_test, y_test];
  }

  private unsignedInt8ToInt32(unsingedInt8Array: Uint8Array): number {
    let res = 0;
    for (let i = 0; i < unsingedInt8Array.length; i++) {
      res = res * 256 + unsingedInt8Array[i];
    }
    return res;
  }

  private async loadImages(filename: string): Promise<Matrix> {
    const images = await Deno.readFile(filename);

    // Bytes 4, 5, 6 & 7 contain number of items
    const itemCount = this.unsignedInt8ToInt32(images.slice(4, 8));

    // Bytes 8, 9, 10 & 11 contain number of rows
    const rowCount = this.unsignedInt8ToInt32(images.slice(8, 12));

    // Bytes 12, 13, 14 & 15 contain number of columns
    const colCount = this.unsignedInt8ToInt32(images.slice(12, 16));

    var pixelValues = [];

    for (let image = 0; image < itemCount; image++) {
      var pixels = [];

      for (let y = 0; y < colCount; y++) {
        for (let x = 0; x < rowCount; x++) {
          // Offset by 16 because first 16 bytes are used for magic number + data size
          // See description of file format on http://yann.lecun.com/exdb/mnist/ for more details
          pixels.push(images[image * rowCount * colCount + (x + y * colCount) + 16]);
        }
      }

      pixelValues.push(pixels);
    }

    return new Matrix(pixelValues);
  }

  private async loadLabels(filename: string): Promise<Matrix> {
    const labels = await Deno.readFile(filename);

    const labelData = Array.from(labels.slice(8));

    return new Matrix([labelData]).transpose();
  }
}
