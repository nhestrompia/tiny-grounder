// mouseclick: post a real left click (move, down, up) at a screen point, in points with a top-left origin.
//
//   mouseclick <x> <y>
//
// System Events' "click at" is ignored by many SwiftUI/Catalyst apps; a HID-level event is what a person's click is.
// Needs the Accessibility permission (the same one axdump uses).
import Foundation
import CoreGraphics
import ApplicationServices

let args = CommandLine.arguments
guard args.count == 3, let x = Double(args[1]), let y = Double(args[2]) else {
    FileHandle.standardError.write("usage: mouseclick <x> <y>\n".data(using: .utf8)!)
    exit(1)
}
guard AXIsProcessTrusted() else {
    FileHandle.standardError.write("mouseclick needs the Accessibility permission\n".data(using: .utf8)!)
    exit(2)
}
let point = CGPoint(x: x, y: y)
let source = CGEventSource(stateID: .hidSystemState)
func post(_ type: CGEventType) {
    CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: point, mouseButton: .left)?
        .post(tap: .cghidEventTap)
}
post(.mouseMoved)
usleep(60_000)
post(.leftMouseDown)
usleep(40_000)
post(.leftMouseUp)
