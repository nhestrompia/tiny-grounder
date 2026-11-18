import Foundation

/// One UI element, matching `grounder/records.py:Element` (JSON keys are the Python field names).
public struct DumpElement: Codable, Equatable {
    public var id: Int
    public var role: String
    public var text: String
    public var context: String
    public var tag: String
    public var box: [Double]
    public var absBox: [Double]
    public var inViewport: Bool
    public var enabled: Bool
    public var conf: Double = 1.0
    public var source: String = "ax"
    public var href: String = ""

    enum CodingKeys: String, CodingKey {
        case id, role, text, context, tag, box, conf, source, href, enabled
        case absBox = "abs_box"
        case inViewport = "in_viewport"
    }

    public init(id: Int, role: String, text: String, context: String, tag: String, box: [Double],
                absBox: [Double], inViewport: Bool, enabled: Bool) {
        self.id = id; self.role = role; self.text = text; self.context = context; self.tag = tag
        self.box = box; self.absBox = absBox; self.inViewport = inViewport; self.enabled = enabled
    }
}

/// The JSON document `axdump` prints: viewport and doc_size are the window size in screen points; `origin` is the
/// window's top-left corner in screen points and `app` the dumped app's name (both extra to the Page fields).
public struct Dump: Codable {
    public var viewport: [Int]
    public var docSize: [Int]
    public var title: String
    public var app: String
    public var origin: [Double]
    public var elements: [DumpElement]

    enum CodingKeys: String, CodingKey {
        case viewport, title, app, origin, elements
        case docSize = "doc_size"
    }

    public init(viewport: [Int], docSize: [Int], title: String, app: String, origin: [Double], elements: [DumpElement]) {
        self.viewport = viewport; self.docSize = docSize; self.title = title; self.app = app
        self.origin = origin; self.elements = elements
    }
}
