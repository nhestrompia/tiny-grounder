// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "axdump",
    platforms: [.macOS(.v13)],
    targets: [
        .target(name: "AXCore"),
        .executableTarget(name: "axdump", dependencies: ["AXCore"]),
        .executableTarget(name: "ocrlines"),
        .executableTarget(name: "mouseclick"),
        .testTarget(name: "AXCoreTests", dependencies: ["AXCore"]),
    ]
)
