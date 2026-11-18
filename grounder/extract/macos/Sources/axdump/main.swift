import AppKit
import ApplicationServices
import AXCore
import Foundation

// axdump: dump the interesting accessibility elements of one app window as JSON.
//   axdump --pid 123 | --app Finder   [--max-depth 40] [--max-elements 2000] [--timeout 10]
// Needs the Accessibility permission for the terminal that runs it. Exit codes: 0 ok, 1 usage, 2 no permission,
// 3 app or window not found.

struct Options {
    var pid: pid_t?
    var appName: String?
    var maxDepth = 40
    var maxElements = 2000
    var timeout = 10.0
}

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

func parseOptions(_ args: [String]) -> Options {
    var opts = Options()
    var i = 0
    func value() -> String {
        i += 1
        guard i < args.count else { fail("missing value for \(args[i - 1])", code: 1) }
        return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--pid": opts.pid = pid_t(value())
        case "--app": opts.appName = value()
        case "--max-depth": opts.maxDepth = Int(value()) ?? opts.maxDepth
        case "--max-elements": opts.maxElements = Int(value()) ?? opts.maxElements
        case "--timeout": opts.timeout = Double(value()) ?? opts.timeout
        default: fail("usage: axdump --pid <pid> | --app <name> [--max-depth N] [--max-elements N] [--timeout SECONDS]", code: 1)
        }
        i += 1
    }
    if (opts.pid == nil) == (opts.appName == nil) { fail("give exactly one of --pid or --app", code: 1) }
    return opts
}

func findPid(_ opts: Options) -> pid_t {
    if let pid = opts.pid { return pid }
    let name = opts.appName!.lowercased()
    let match = NSWorkspace.shared.runningApplications.first {
        $0.localizedName?.lowercased() == name || $0.bundleIdentifier?.lowercased() == name
    }
    guard let app = match else { fail("no running app named '\(opts.appName!)'", code: 3) }
    return app.processIdentifier
}

// MARK: attribute access

func attribute(_ el: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(el, name as CFString, &value) == .success ? value : nil
}

func string(_ el: AXUIElement, _ name: String) -> String? { attribute(el, name) as? String }

func bool(_ el: AXUIElement, _ name: String) -> Bool? { (attribute(el, name) as? NSNumber)?.boolValue }

func children(_ el: AXUIElement) -> [AXUIElement] { (attribute(el, kAXChildrenAttribute) as? [AXUIElement]) ?? [] }

func actions(_ el: AXUIElement) -> [String] {
    var names: CFArray?
    return AXUIElementCopyActionNames(el, &names) == .success ? (names as? [String]) ?? [] : []
}

/// [x, y, w, h] in screen points.
func frame(_ el: AXUIElement) -> [Double]? {
    var point = CGPoint.zero, size = CGSize.zero
    guard let p = attribute(el, kAXPositionAttribute), let s = attribute(el, kAXSizeAttribute),
          AXValueGetValue(p as! AXValue, .cgPoint, &point), AXValueGetValue(s as! AXValue, .cgSize, &size)
    else { return nil }
    return [Double(point.x), Double(point.y), Double(size.width), Double(size.height)]
}

func pickWindow(_ app: AXUIElement) -> AXUIElement? {
    if let focused = attribute(app, kAXFocusedWindowAttribute) { return (focused as! AXUIElement) }
    return (attribute(app, kAXWindowsAttribute) as? [AXUIElement])?.first
}

// MARK: traversal

struct Walker {
    let window: [Double]
    let windowTitle: String
    let opts: Options
    let deadline: Date
    var elements: [DumpElement] = []

    var done: Bool { elements.count >= opts.maxElements || Date() > deadline }

    /// Text of the static-text children of `parent`, used as sibling context.
    func siblingTexts(_ kids: [AXUIElement]) -> [String] {
        kids.compactMap { kid in
            string(kid, kAXRoleAttribute) == "AXStaticText" ? (string(kid, kAXValueAttribute) ?? string(kid, kAXTitleAttribute)) : nil
        }
    }

    mutating func visit(_ el: AXUIElement, parentRole: String?, groupTitle: String, siblings: [String], depth: Int) {
        if done || depth > opts.maxDepth { return }
        let axRole = string(el, kAXRoleAttribute) ?? ""
        let acts = actions(el)
        if isKept(axRole: axRole, actions: acts), let role = mapRole(axRole, parentRole: parentRole, actions: acts),
           let rect = frame(el), isVisible(rect, window: window) {
            let secure = string(el, kAXSubroleAttribute) == "AXSecureTextField"
            let text = pickText(
                title: string(el, kAXTitleAttribute), description: string(el, kAXDescriptionAttribute),
                value: string(el, kAXValueAttribute), help: string(el, kAXHelpAttribute),
                placeholder: string(el, "AXPlaceholderValue"), isSecure: secure)
            elements.append(DumpElement(
                id: elements.count, role: role, text: text,
                context: buildContext(windowTitle: windowTitle, groupTitle: groupTitle, siblingTexts: siblings, ownText: text),
                tag: axRole, box: normalisedBox(rect, window: window), absBox: rect,
                inViewport: true, enabled: bool(el, kAXEnabledAttribute) ?? true))
        }
        let kids = children(el)
        var nextGroup = groupTitle
        if ["AXGroup", "AXToolbar", "AXScrollArea", "AXTabGroup", "AXSplitGroup"].contains(axRole) {
            let title = collapse(string(el, kAXTitleAttribute) ?? string(el, kAXDescriptionAttribute) ?? "", limit: 40)
            if !title.isEmpty { nextGroup = title }
        }
        let kidTexts = siblingTexts(kids)
        for kid in kids { visit(kid, parentRole: axRole, groupTitle: nextGroup, siblings: kidTexts, depth: depth + 1) }
    }
}

// MARK: main

let opts = parseOptions(Array(CommandLine.arguments.dropFirst()))
guard AXIsProcessTrusted() else {
    fail("""
    axdump needs the Accessibility permission. Open System Settings > Privacy & Security > Accessibility, \
    enable the terminal app you run this from, then run axdump again.
    """, code: 2)
}
let pid = findPid(opts)
let app = AXUIElementCreateApplication(pid)
AXUIElementSetMessagingTimeout(app, 1.0)
guard let window = pickWindow(app), let windowFrame = frame(window) else {
    fail("pid \(pid) has no accessible window (is it running, on screen and not minimised?)", code: 3)
}
let title = collapse(string(window, kAXTitleAttribute) ?? "", limit: maxContextChars)
var walker = Walker(window: windowFrame, windowTitle: title, opts: opts, deadline: Date().addingTimeInterval(opts.timeout))
walker.visit(window, parentRole: nil, groupTitle: "", siblings: [], depth: 0)
let size = [Int(windowFrame[2]), Int(windowFrame[3])]
let dump = Dump(viewport: size, docSize: size, title: title, elements: walker.elements)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
FileHandle.standardOutput.write(try encoder.encode(dump))
FileHandle.standardOutput.write(Data("\n".utf8))
