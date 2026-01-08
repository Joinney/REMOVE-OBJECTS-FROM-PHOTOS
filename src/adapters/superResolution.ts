/* eslint-disable no-console */
/* eslint-disable no-plusplus */
import * as ort from 'onnxruntime-web';
import cv, { Mat } from 'opencv-ts';
import { getCapabilities } from './util';
import { ensureModel } from './cache';

// --- Helper Functions ---

function loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load image from ${url}`));
        img.src = url;
    });
}

function imgProcess(img: Mat) {
    const channels = new cv.MatVector();
    cv.split(img, channels); // Tách kênh màu

    const C = channels.size();
    const H = img.rows;
    const W = img.cols;

    const chwArray = new Float32Array(C * H * W);

    for (let c = 0; c < C; c++) {
        const channelData = channels.get(c).data;
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                // Chuẩn hóa về [0, 1]
                chwArray[c * H * W + h * W + w] = channelData[h * W + w] / 255.0;
            }
        }
    }

    channels.delete();
    return chwArray;
}

// --- Core Logic ---

async function tileProc(
    inputTensor: ort.Tensor,
    session: ort.InferenceSession,
    callback: (progress: number) => void
) {
    const inputDims = inputTensor.dims;

    // FIX TS2322: Ép kiểu dims về number (Dòng 116-118 cũ)
    const imageW = inputDims[3] as number;
    const imageH = inputDims[2] as number;

    // Nếu cần channel input: const imageC = inputDims[1] as number;

    const rOffset = 0;
    const gOffset = imageW * imageH;
    const bOffset = imageW * imageH * 2;

    // FIX TS2322: Ép kiểu từng phần tử khi tạo mảng outputDims
    const outputDims = [
        inputDims[0] as number,
        inputDims[1] as number,
        (inputDims[2] as number) * 4,
        (inputDims[3] as number) * 4,
    ];

    const outputTensor = new ort.Tensor(
        'float32',
        new Float32Array(
            outputDims[0] * outputDims[1] * outputDims[2] * outputDims[3]
        ),
        outputDims
    );

    // FIX TS2322: Ép kiểu về number cho các biến output (Dòng 151-155 cũ)
    const outImageW = outputDims[3] as number;
    const outImageH = outputDims[2] as number;
    // const outImageC = outputDims[1] as number; // Nếu cần

    const outROffset = 0;
    // Tính offset dựa trên kích thước đã ép kiểu
    const outGOffset = outImageW * outImageH;
    const outBOffset = outImageW * outImageH * 2;

    const tileSize = 64;
    const tilePadding = 6;
    const tileSizePre = tileSize - tilePadding * 2;

    const tilesx = Math.ceil(imageW / tileSizePre);
    const tilesy = Math.ceil(imageH / tileSizePre);

    // Ép kiểu data đầu vào
    const data = inputTensor.data as Float32Array;

    console.log(inputTensor);
    const numTiles = tilesx * tilesy;
    let currentTile = 0;

    // Ép kiểu data đầu ra để gán giá trị
    const outputData = outputTensor.data as Float32Array;

    for (let i = 0; i < tilesx; i++) {
        for (let j = 0; j < tilesy; j++) {
            const ti = Date.now();
            const tileW = Math.min(tileSizePre, imageW - i * tileSizePre);
            const tileH = Math.min(tileSizePre, imageH - j * tileSizePre);

            const tileROffset = 0;
            const tileGOffset = tileSize * tileSize;
            const tileBOffset = tileSize * tileSize * 2;

            const tileData = new Float32Array(tileSize * tileSize * 3);

            for (let xp = -tilePadding; xp < tileSizePre + tilePadding; xp++) {
                for (let yp = -tilePadding; yp < tileSizePre + tilePadding; yp++) {
                    let xim = i * tileSizePre + xp;
                    if (xim < 0) xim = 0;
                    else if (xim >= imageW) xim = imageW - 1;

                    let yim = j * tileSizePre + yp;
                    if (yim < 0) yim = 0;
                    else if (yim >= imageH) yim = imageH - 1;

                    const idx = xim + yim * imageW;
                    const xt = xp + tilePadding;
                    const yt = yp + tilePadding;

                    tileData[xt + yt * tileSize + tileROffset] = data[idx + rOffset];
                    tileData[xt + yt * tileSize + tileGOffset] = data[idx + gOffset];
                    tileData[xt + yt * tileSize + tileBOffset] = data[idx + bOffset];
                }
            }

            const tile = new ort.Tensor('float32', tileData, [1, 3, tileSize, tileSize]);

            // Chạy model
            const r = await session.run({ 'input.1': tile });
            // Lấy output key đầu tiên (thường là dynamic key)
            const outputKey = Object.keys(r)[0];
            const results = { output: r[outputKey] };

            const outTileW = tileW * 4;
            const outTileH = tileH * 4;
            const outTileSize = tileSize * 4;
            const outTileSizePre = tileSizePre * 4;

            const outTileROffset = 0;
            const outTileGOffset = outTileSize * outTileSize;
            const outTileBOffset = outTileSize * outTileSize * 2;

            const resultData = results.output.data as Float32Array;

            for (let x = 0; x < outTileW; x++) {
                for (let y = 0; y < outTileH; y++) {
                    const xim = i * outTileSizePre + x;
                    const yim = j * outTileSizePre + y;
                    const idx = xim + yim * outImageW;

                    const xt = x + tilePadding * 4;
                    const yt = y + tilePadding * 4;

                    outputData[idx + outROffset] = resultData[xt + yt * outTileSize + outTileROffset];
                    outputData[idx + outGOffset] = resultData[xt + yt * outTileSize + outTileGOffset];
                    outputData[idx + outBOffset] = resultData[xt + yt * outTileSize + outTileBOffset];
                }
            }

            currentTile++;
            const dt = Date.now() - ti;
            const remTime = (numTiles - currentTile) * dt;
            console.log(
                `tile ${currentTile} of ${numTiles} took ${dt} ms, remaining time: ${remTime} ms`
            );
            callback(Math.round(100 * (currentTile / numTiles)));
        }
    }

    console.log(`output dims:${outputTensor.dims}`);
    return outputTensor;
}

function processImage(
    img: HTMLImageElement,
    canvasId?: string
): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
        try {
            const src = cv.imread(img);
            const src_rgb = new cv.Mat();
            cv.cvtColor(src, src_rgb, cv.COLOR_RGBA2RGB);
            if (canvasId) {
                cv.imshow(canvasId, src_rgb);
            }
            resolve(imgProcess(src_rgb));

            src.delete();
            src_rgb.delete();
        } catch (error) {
            reject(error);
        }
    });
}

function configEnv(capabilities: {
    webgpu: any;
    wasm?: boolean;
    simd: any;
    threads: any;
}) {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
    if (capabilities.webgpu) {
        ort.env.wasm.numThreads = 1;
    } else {
        if (capabilities.threads) {
            ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
        }
        if (capabilities.simd) {
            ort.env.wasm.simd = true;
        }
        ort.env.wasm.proxy = true;
    }
    console.log('env', ort.env.wasm);
}

function postProcess(floatData: Float32Array, width: number, height: number) {
    const chwToHwcData = [];
    const size = width * height;

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            for (let c = 0; c < 3; c++) {
                const chwIndex = c * size + h * width + w;
                const pixelVal = floatData[chwIndex];
                let newPiex = pixelVal;
                if (pixelVal > 1) {
                    newPiex = 1;
                } else if (pixelVal < 0) {
                    newPiex = 0;
                }
                chwToHwcData.push(newPiex * 255);
            }
            chwToHwcData.push(255); // Alpha channel
        }
    }
    return chwToHwcData;
}

function imageDataToDataURL(imageData: ImageData) {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Could not get 2d context from canvas');
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
}

// Biến lưu session
let session: ort.InferenceSession | null = null;

export default async function superResolution(
    imageFile: File | HTMLImageElement,
    callback: (progress: number) => void
) {
    console.time('sessionCreate');

    if (!session) {
        const capabilities = await getCapabilities();
        configEnv(capabilities);

        // Lấy model buffer
        const modelBuffer = await ensureModel('superResolution');

        // Tạo session từ modelBuffer
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: [capabilities.webgpu ? 'webgpu' : 'wasm'],
        });
    }
    console.timeEnd('sessionCreate');

    const img =
        imageFile instanceof HTMLImageElement
            ? imageFile
            : await loadImage(URL.createObjectURL(imageFile));

    const imageTersorData = await processImage(img);

    const imageTensor = new ort.Tensor('float32', imageTersorData, [
        1,
        3,
        img.height,
        img.width,
    ]);

    // Kiểm tra session trước khi dùng
    if (!session) throw new Error("Session creation failed");

    const result = await tileProc(imageTensor, session, callback);

    console.time('postProcess');
    const outsTensor = result;

    // Ép kiểu data thành Float32Array
    const chwToHwcData = postProcess(
        outsTensor.data as Float32Array,
        img.width * 4,
        img.height * 4
    );

    const imageData = new ImageData(
        new Uint8ClampedArray(chwToHwcData),
        img.width * 4,
        img.height * 4
    );

    console.log(imageData, 'imageData');
    const url = imageDataToDataURL(imageData);
    console.timeEnd('postProcess');

    return url;
}