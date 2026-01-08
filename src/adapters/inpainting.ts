/* eslint-disable camelcase */
/* eslint-disable no-plusplus */
/* eslint-disable no-console */
import * as ort from 'onnxruntime-web'
import cv, { Mat } from 'opencv-ts'
import { ensureModel } from './cache'
import { getCapabilities } from './util'

// ort.env.debug = true
// ort.env.logLevel = 'verbose'
// ort.env.webgpu.profilingMode = 'default'

function loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image()
        img.crossOrigin = 'Anonymous'
        img.onload = () => resolve(img)
        img.onerror = () => reject(new Error(`Failed to load image from ${url}`))
        img.src = url
    })
}

function imgProcess(img: Mat) {
    const channels = new cv.MatVector()
    cv.split(img, channels) // Phân tách kênh màu

    const C = channels.size() // Số kênh
    const H = img.rows // Chiều cao
    const W = img.cols // Chiều rộng

    const chwArray = new Uint8Array(C * H * W) // Mảng lưu dữ liệu chuyển đổi

    for (let c = 0; c < C; c++) {
        const channelData = channels.get(c).data // Lấy dữ liệu từng kênh
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                chwArray[c * H * W + h * W + w] = channelData[h * W + w]
            }
        }
    }

    channels.delete() // Giải phóng bộ nhớ
    return chwArray
}

function markProcess(img: Mat) {
    const channels = new cv.MatVector()
    cv.split(img, channels) // Phân tách kênh màu

    const C = 1 // Số kênh (Mask chỉ có 1)
    const H = img.rows
    const W = img.cols

    const chwArray = new Uint8Array(C * H * W)

    for (let c = 0; c < C; c++) {
        const channelData = channels.get(0).data
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                // Fix lỗi Prettier: ngắt dòng và ép kiểu rõ ràng
                chwArray[c * H * W + h * W + w] =
                    ((channelData[h * W + w] !== 255) as unknown as number) * 255
            }
        }
    }

    channels.delete()
    return chwArray
}

function processImage(
    img: HTMLImageElement,
    canvasId?: string
): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
        try {
            const src = cv.imread(img)
            const src_rgb = new cv.Mat()
            // Chuyển từ RGBA sang RGB
            cv.cvtColor(src, src_rgb, cv.COLOR_RGBA2RGB)
            if (canvasId) {
                cv.imshow(canvasId, src_rgb)
            }
            resolve(imgProcess(src_rgb))

            src.delete()
            src_rgb.delete()
        } catch (error) {
            reject(error)
        }
    })
}

function processMark(
    img: HTMLImageElement,
    canvasId?: string
): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
        try {
            const src = cv.imread(img)
            const src_grey = new cv.Mat()

            // Chuyển từ RGBA sang Grayscale (nhị phân)
            cv.cvtColor(src, src_grey, cv.COLOR_BGR2GRAY)

            if (canvasId) {
                cv.imshow(canvasId, src_grey)
            }

            resolve(markProcess(src_grey))

            src.delete()
        } catch (error) {
            reject(error)
        }
    })
}

function postProcess(uint8Data: Uint8Array, width: number, height: number) {
    const chwToHwcData = []
    const size = width * height

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            for (let c = 0; c < 3; c++) {
                // RGB channels
                const chwIndex = c * size + h * width + w
                const pixelVal = uint8Data[chwIndex]
                let newPiex = pixelVal
                if (pixelVal > 255) {
                    newPiex = 255
                } else if (pixelVal < 0) {
                    newPiex = 0
                }
                chwToHwcData.push(newPiex)
            }
            chwToHwcData.push(255) // Alpha channel
        }
    }
    return chwToHwcData
}

function imageDataToDataURL(imageData: ImageData) {
    // Tạo canvas
    const canvas = document.createElement('canvas')
    canvas.width = imageData.width
    canvas.height = imageData.height

    // Vẽ imageData lên canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) {
        throw new Error('Could not get 2d context from canvas')
    }
    ctx.putImageData(imageData, 0, 0)

    // Xuất ra Data URL
    return canvas.toDataURL()
}

function configEnv(capabilities: {
    webgpu: any
    wasm?: boolean
    simd: any
    threads: any
}) {
    ort.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/'
    if (capabilities.webgpu) {
        ort.env.wasm.numThreads = 1
    } else {
        if (capabilities.threads) {
            ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4
        }
        if (capabilities.simd) {
            ort.env.wasm.simd = true
        }
        ort.env.wasm.proxy = true
    }
    console.log('env', ort.env.wasm)
}

const resizeMark = (
    image: HTMLImageElement,
    width: number,
    height: number
): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height

        // Vẽ và resize ảnh trên canvas
        const ctx = canvas.getContext('2d')
        if (!ctx) {
            reject(new Error('Unable to get canvas context'))
            return
        }
        ctx.drawImage(image, 0, 0, width, height)

        const resizedImageUrl = canvas.toDataURL()

        const resizedImage = new Image()
        resizedImage.onload = () => resolve(resizedImage)
        resizedImage.onerror = () =>
            reject(new Error('Failed to load resized image'))
        resizedImage.src = resizedImageUrl
    })
}

let model: ort.InferenceSession | null = null

export default async function inpaint(
    imageFile: File | HTMLImageElement,
    maskBase64: string
) {
    console.time('sessionCreate')
    if (!model) {
        const capabilities = await getCapabilities()
        configEnv(capabilities)
        const modelBuffer = await ensureModel('inpaint')
        model = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: [capabilities.webgpu ? 'webgpu' : 'wasm'],
        })
    }
    console.timeEnd('sessionCreate')
    console.time('preProcess')

    const [originalImg, originalMark] = await Promise.all([
        imageFile instanceof HTMLImageElement
            ? imageFile
            : loadImage(URL.createObjectURL(imageFile)),
        loadImage(maskBase64),
    ])

    const [img, mark] = await Promise.all([
        processImage(originalImg),
        processMark(
            await resizeMark(originalMark, originalImg.width, originalImg.height)
        ),
    ])

    const imageTensor = new ort.Tensor('uint8', img, [
        1,
        3,
        originalImg.height,
        originalImg.width,
    ])

    const maskTensor = new ort.Tensor('uint8', mark, [
        1,
        1,
        originalImg.height,
        originalImg.width,
    ])

    if (!model) {
        throw new Error('Model failed to initialize')
    }

    const Feed: {
        [key: string]: any
    } = {
        [model.inputNames[0]]: imageTensor,
        [model.inputNames[1]]: maskTensor,
    }

    console.timeEnd('preProcess')

    console.time('run')
    const results = await model.run(Feed)
    console.timeEnd('run')

    console.time('postProcess')
    const outsTensor = results[model.outputNames[0]]

    // Ép kiểu as Uint8Array để fix lỗi TS2345
    const chwToHwcData = postProcess(
        outsTensor.data as Uint8Array,
        originalImg.width,
        originalImg.height
    )

    const imageData = new ImageData(
        new Uint8ClampedArray(chwToHwcData),
        originalImg.width,
        originalImg.height
    )
    console.log(imageData, 'imageData')
    const result = imageDataToDataURL(imageData)
    console.timeEnd('postProcess')

    return result
}