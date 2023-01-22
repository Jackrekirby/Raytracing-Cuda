const fs = require('fs');
const canvas = require('canvas');


function create_png(infile, outfile, width, height) {
    let pixels = new Uint8ClampedArray(fs.readFileSync(infile + '.bin', null));
    const imageData = new canvas.ImageData(width, height);

    let j = 0;
    const size = width * height * 3;
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
    fs.writeFileSync(outfile + '.png', buffer);
}


let [_node_exe, _main_js, infile, outfile, width, height] = process.argv;


const dir = outfile.split('/').slice(0, -1).join('/');

fs.readdir(dir, (_err, files) => {
    const nfiles = String(999 - files.length).padStart(3, '0');
    outfile = outfile + '_' + nfiles;
    console.log("bin2png", infile, outfile, width, height);
    create_png(infile, outfile, Number(width), Number(height));
});





