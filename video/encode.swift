// Encode a folder of PNG frames and an AAC audio file into an H.264 MP4 with AVFoundation,
// so the video needs no ffmpeg. Usage: encode <frames dir> <fps> <audio.m4a> <out.mp4>
import AVFoundation
import CoreGraphics
import ImageIO

let args = CommandLine.arguments
let dir = URL(fileURLWithPath: args[1])
let fps = Int32(args[2])!
let audioURL = URL(fileURLWithPath: args[3])
let outURL = URL(fileURLWithPath: args[4])
let tmpURL = outURL.deletingPathExtension().appendingPathExtension("silent.mov")
try? FileManager.default.removeItem(at: tmpURL)
try? FileManager.default.removeItem(at: outURL)

let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".png") }.sorted()
func load(_ name: String) -> CGImage {
    let src = CGImageSourceCreateWithURL(dir.appendingPathComponent(name) as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(src, 0, nil)!
}
let first = load(files[0])
let w = first.width, h = first.height

let writer = try AVAssetWriter(outputURL: tmpURL, fileType: .mov)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
    AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 10_000_000,
                                      AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel]])
input.expectsMediaDataInRealTime = false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h])
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)
for (i, f) in files.enumerated() {
    while !input.isReadyForMoreMediaData { usleep(1000) }
    let img = load(f)
    var pb: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
    CVPixelBufferLockBaseAddress(pb!, [])
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb!), width: w, height: h, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb!), space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    CVPixelBufferUnlockBaseAddress(pb!, [])
    adaptor.append(pb!, withPresentationTime: CMTime(value: Int64(i), timescale: fps))
}
input.markAsFinished()
let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()

let comp = AVMutableComposition()
let vAsset = AVURLAsset(url: tmpURL), aAsset = AVURLAsset(url: audioURL)
let vTrack = try await vAsset.loadTracks(withMediaType: .video)[0]
let aTrack = try await aAsset.loadTracks(withMediaType: .audio)[0]
let dur = try await vAsset.load(.duration)
let aDur = try await aAsset.load(.duration)
try comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)!
    .insertTimeRange(CMTimeRange(start: .zero, duration: dur), of: vTrack, at: .zero)
try comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
    .insertTimeRange(CMTimeRange(start: .zero, duration: CMTimeMinimum(dur, aDur)), of: aTrack, at: .zero)
let export = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetPassthrough)!
try await export.export(to: outURL, as: .mp4)
try? FileManager.default.removeItem(at: tmpURL)
print("wrote \(outURL.path) (\(files.count) frames)")
