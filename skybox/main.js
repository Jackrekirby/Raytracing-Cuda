const fs = require("fs");
const pngjs = require("pngjs");
const canvas = require("canvas");

console.log('Skybox Importer');

const infile = 'skybox2.png';
// const infile = 'skybox_debug.png';

fs.createReadStream(infile)
    .pipe(
        new pngjs.PNG({
            filterType: 4,
        })
    )
    .on("parsed", read_image);

function read_image() {
    const width = this.width, height = this.height;
    console.log('image size', this.width, this.height);

    const isInteger = (number) => number % 1 == 0;
    if (!(isInteger(width / 4) && isInteger(height / 3) && (width / 4) === (height / 3))) return;

    const faceSize = width / 4;
    console.log('face size', faceSize)

    const imap = [
        { x: 2, y: 1 },
        { x: 0, y: 1 },
        { x: 1, y: 0 },
        { x: 1, y: 2 },
        { x: 1, y: 1 },
        { x: 3, y: 1 },
    ]


    let k = 0;

    const imageData = new canvas.ImageData(faceSize, faceSize * 6);

    let pixels = new Uint8ClampedArray(faceSize * faceSize * 6 * 3);

    let ip = 0;
    for (const z of imap) {
        for (let y = 0; y < faceSize; y++) {
            for (let x = 0; x < faceSize; x++) {
                const i = (x + y * faceSize + (k * faceSize * faceSize)) * 4;
                const j = ((z.x * faceSize + x) + (z.y * faceSize + y) * width) * 4

                imageData.data[i] = this.data[j];
                imageData.data[i + 1] = this.data[j + 1];
                imageData.data[i + 2] = this.data[j + 2];
                imageData.data[i + 3] = this.data[j + 3];


                pixels[ip] = this.data[j];
                pixels[ip + 1] = this.data[j + 1];
                pixels[ip + 2] = this.data[j + 2];
                ip += 3;
            }
        }


        k++;
    }

    const image = canvas.createCanvas(faceSize, faceSize * 6);
    const context = image.getContext("2d");

    context.putImageData(imageData, 0, 0);

    // Write the image to file
    const buffer = image.toBuffer("image/png");
    fs.writeFileSync(`./skybox_.png`, buffer);

    // this.pack().pipe(fs.createWriteStream("out.png"));

    fs.writeFileSync("skybox.bin", pixels, "binary", err => err ? err : "The file was saved!");


    create_png("skybox.bin", "skybox_bin.png", faceSize, faceSize * 6);

}


function create_png(infile, outfile, width, height) {
    let pixels = new Uint8ClampedArray(fs.readFileSync(infile, null));
    console.log(pixels);

    const imageData = new canvas.ImageData(width, height);

    j = 0;
    size = width * height * 3;
    for (let i = 0; i < size; i += 3) {
        imageData.data[j] = pixels[i];
        imageData.data[j + 1] = pixels[i + 1];
        imageData.data[j + 2] = pixels[i + 2];
        imageData.data[j + 3] = 255;
        j += 4;
    }

    const image = canvas.createCanvas(width, height);
    const context = image.getContext("2d");

    context.putImageData(imageData, 0, 0);

    const buffer = image.toBuffer("image/png");
    fs.writeFileSync(outfile, buffer);
}
