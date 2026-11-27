// ocrlines: recognise text lines in a PNG with Apple's on-device Vision framework.
//
//   ocrlines <image.png>
//
// Prints JSON {"width": W, "height": H, "lines": [{"text": "...", "box": [x, y, w, h]}]} with boxes in image
// pixels, origin top-left. Used to redact personal text from collector screenshots; nothing leaves the machine.
import Foundation
import Vision
import ImageIO

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

let args = CommandLine.arguments
guard args.count == 2 else { fail("usage: ocrlines <image.png>", code: 1) }
let url = URL(fileURLWithPath: args[1])
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { fail("cannot read \(args[1])", code: 2) }
let width = Double(image.width), height = Double(image.height)

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
do {
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
} catch {
    fail("recognition failed: \(error)", code: 3)
}
var lines: [[String: Any]] = []
for observation in request.results ?? [] {
    guard let candidate = observation.topCandidates(1).first else { continue }
    let b = observation.boundingBox  // normalised, origin bottom-left
    lines.append(["text": candidate.string,
                  "box": [b.minX * width, (1 - b.maxY) * height, b.width * width, b.height * height]])
}
let out: [String: Any] = ["width": Int(width), "height": Int(height), "lines": lines]
let data = try! JSONSerialization.data(withJSONObject: out, options: [])
FileHandle.standardOutput.write(data)
