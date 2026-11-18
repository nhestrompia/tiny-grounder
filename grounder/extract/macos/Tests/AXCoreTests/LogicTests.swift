import XCTest
@testable import AXCore

final class LogicTests: XCTestCase {
    func testRoleMap() {
        XCTAssertEqual(mapRole("AXButton", parentRole: nil, actions: []), "button")
        XCTAssertEqual(mapRole("AXMenuButton", parentRole: nil, actions: []), "button")
        XCTAssertEqual(mapRole("AXIncrementor", parentRole: nil, actions: []), "slider")
        XCTAssertEqual(mapRole("AXDisclosureTriangle", parentRole: nil, actions: []), "button")
        XCTAssertEqual(mapRole("AXSearchField", parentRole: nil, actions: []), "searchbox")
        XCTAssertEqual(mapRole("AXRadioButton", parentRole: "AXTabGroup", actions: []), "tab")
        XCTAssertEqual(mapRole("AXRadioButton", parentRole: "AXGroup", actions: []), "radio")
    }

    func testContainersNeedAnAction() {
        XCTAssertNil(mapRole("AXCell", parentRole: nil, actions: []))
        XCTAssertNil(mapRole("AXRow", parentRole: nil, actions: ["AXScrollToVisible"]))
        XCTAssertEqual(mapRole("AXRow", parentRole: nil, actions: ["AXPick"]), "other")
        XCTAssertEqual(mapRole("AXGroup", parentRole: nil, actions: ["AXPress"]), "button")
        XCTAssertNil(mapRole("AXGroup", parentRole: nil, actions: []))
        XCTAssertNil(mapRole("AXScrollBar", parentRole: nil, actions: []))
    }

    func testKept() {
        XCTAssertTrue(isKept(axRole: "AXTextField", actions: []))
        XCTAssertTrue(isKept(axRole: "AXImage", actions: ["AXPress"]))
        XCTAssertFalse(isKept(axRole: "AXStaticText", actions: []))
        XCTAssertFalse(isKept(axRole: "AXButton", actions: ["AXShowDefaultUI"]))
    }

    func testTextPriorityAndSecureFields() {
        XCTAssertEqual(pickText(title: "", description: " Save  file ", value: "v", help: "h", placeholder: "p", isSecure: false), "Save file")
        XCTAssertEqual(pickText(title: nil, description: nil, value: "typed", help: "h", placeholder: nil, isSecure: false), "typed")
        XCTAssertEqual(pickText(title: nil, description: nil, value: "hunter2", help: nil, placeholder: "Password", isSecure: true), "Password")
        XCTAssertEqual(pickText(title: nil, description: nil, value: nil, help: nil, placeholder: nil, isSecure: false), "")
        XCTAssertEqual(pickText(title: String(repeating: "a", count: 200), description: nil, value: nil, help: nil, placeholder: nil, isSecure: false).count, 80)
    }

    func testContext() {
        XCTAssertEqual(buildContext(windowTitle: "Settings", groupTitle: "Network", siblingTexts: ["Wi-Fi", "Save"], ownText: "Save"),
                       "Settings · Network · Wi-Fi")
        XCTAssertEqual(buildContext(windowTitle: "", groupTitle: "", siblingTexts: [], ownText: "x"), "")
        XCTAssertLessThanOrEqual(buildContext(windowTitle: String(repeating: "w", count: 300), groupTitle: "g", siblingTexts: [], ownText: "").count, 120)
    }

    func testBoxAndVisibility() {
        let window = [100.0, 50.0, 800.0, 600.0]
        XCTAssertEqual(normalisedBox([500, 350, 80, 30], window: window), [0.5, 0.5, 0.1, 0.05])
        XCTAssertTrue(isVisible([500, 350, 80, 30], window: window))
        XCTAssertTrue(isVisible([850, 100, 80, 30], window: window))
        XCTAssertFalse(isVisible([950, 100, 80, 30], window: window))
        XCTAssertFalse(isVisible([500, 350, 0, 30], window: window))
    }

    func testDumpJSONUsesRecordFieldNames() throws {
        let el = DumpElement(id: 1, role: "button", text: "OK", context: "", tag: "AXButton", box: [0, 0, 1, 1],
                             absBox: [1, 2, 3, 4], inViewport: true, enabled: true)
        let data = try JSONEncoder().encode(Dump(viewport: [800, 600], docSize: [800, 600], title: "T", app: "Notes",
                                                         origin: [10, 20], elements: [el]))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNotNil(json["doc_size"])
        XCTAssertEqual(json["app"] as? String, "Notes")
        XCTAssertEqual(json["origin"] as? [Double], [10, 20])
        let first = try XCTUnwrap((json["elements"] as? [[String: Any]])?.first)
        XCTAssertEqual(first["abs_box"] as? [Double], [1, 2, 3, 4])
        XCTAssertEqual(first["source"] as? String, "ax")
        XCTAssertEqual(first["in_viewport"] as? Bool, true)
    }
}
