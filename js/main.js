const fs = require('fs');
const canvas = require("canvas");
require('log-timestamp');

const pixel_file = '../scene.bin';
let T = 0;

console.log(`Watching for file changes on ${pixel_file}`);

function create_png() {
    let pixels = new Uint8ClampedArray(fs.readFileSync(pixel_file, null));
    // console.log(pixels);

    width = 1920;
    height = 1080;

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

    // console.log(imageData);

    // 8,294,400 6,220,800

    // Instantiate the canvas object
    const image = canvas.createCanvas(width, height);
    const context = image.getContext("2d");

    context.putImageData(imageData, 0, 0);

    // Write the image to file
    const buffer = image.toBuffer("image/png");
    fs.writeFileSync(`../img/scene_${T}.png`, buffer);
    T += 1;
}

create_png()

fs.watchFile(pixel_file, (curr, prev) => {
    console.log(`${pixel_file} file changed`);

    create_png();

    console.log(`${pixel_file} updated`);
});

