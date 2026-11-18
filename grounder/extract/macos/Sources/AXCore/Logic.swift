import Foundation

public let maxTextChars = 80
public let maxContextChars = 120

/// Actions that make an accessibility element worth keeping.
public let actionableActions: Set<String> = ["AXPress", "AXConfirm", "AXPick", "AXShowMenu", "AXIncrement"]
/// Roles that take input and are kept even without an action.
public let editableRoles: Set<String> = ["AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXSlider"]

private let roleMap: [String: String] = [
    "AXButton": "button", "AXLink": "link", "AXTextField": "textbox", "AXTextArea": "textbox",
    "AXSearchField": "searchbox", "AXCheckBox": "checkbox", "AXRadioButton": "radio",
    "AXPopUpButton": "combobox", "AXComboBox": "combobox", "AXMenuItem": "menuitem",
    "AXMenuButton": "button", "AXSlider": "slider", "AXIncrementor": "slider",
    "AXDisclosureTriangle": "button", "AXStaticText": "text", "AXImage": "image",
]
/// Container roles that only count when something can be done with them.
private let containerRoles: Set<String> = ["AXGroup", "AXCell", "AXRow"]

public func isActionable(_ actions: [String]) -> Bool {
    actions.contains { actionableActions.contains($0) }
}

/// The record role for an AX role, or nil when the element is not interesting.
/// `parentRole` turns radio buttons inside a tab group into tabs; groups, cells and rows are
/// kept (as button / other) only when they have an action.
public func mapRole(_ axRole: String, parentRole: String?, actions: [String]) -> String? {
    if axRole == "AXRadioButton" && parentRole == "AXTabGroup" { return "tab" }
    if containerRoles.contains(axRole) {
        guard isActionable(actions) else { return nil }
        return axRole == "AXGroup" && actions.contains("AXPress") ? "button" : "other"
    }
    if let role = roleMap[axRole] { return role }
    return isActionable(actions) ? "other" : nil
}

/// Keep an element when it has an action or takes input. Plain text and images are context only.
public func isKept(axRole: String, actions: [String]) -> Bool {
    isActionable(actions) || editableRoles.contains(axRole)
}

public func collapse(_ s: String, limit: Int) -> String {
    let flat = s.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    return String(flat.prefix(limit))
}

/// First non-empty of title, description, value, help, placeholder (in that order).
/// `value` is used only when it is a string and the field is not a secure text field.
public func pickText(title: String?, description: String?, value: String?, help: String?,
                     placeholder: String?, isSecure: Bool) -> String {
    let candidates = [title, description, isSecure ? nil : value, help, placeholder]
    for candidate in candidates {
        let text = collapse(candidate ?? "", limit: maxTextChars)
        if !text.isEmpty { return text }
    }
    return ""
}

/// Context: window title, nearest titled ancestor group, sibling static texts; own text removed; at most 120 chars.
public func buildContext(windowTitle: String, groupTitle: String, siblingTexts: [String], ownText: String) -> String {
    let siblings = siblingTexts.map { collapse($0, limit: maxTextChars) }.filter { !$0.isEmpty && $0 != ownText }
    let parts = [windowTitle, groupTitle, siblings.joined(separator: " ")]
        .map { collapse($0, limit: maxContextChars) }
        .filter { !$0.isEmpty && $0 != ownText }
    return String(parts.joined(separator: " · ").prefix(maxContextChars))
}

/// [x/ww, y/wh, w/ww, h/wh] relative to the window; screen-point rects are [x, y, w, h].
public func normalisedBox(_ rect: [Double], window: [Double]) -> [Double] {
    let (wx, wy, ww, wh) = (window[0], window[1], window[2], window[3])
    func r(_ v: Double) -> Double { (v * 10_000).rounded() / 10_000 }
    return [r((rect[0] - wx) / ww), r((rect[1] - wy) / wh), r(rect[2] / ww), r(rect[3] / wh)]
}

/// True when the rect has area and overlaps the window frame.
public func isVisible(_ rect: [Double], window: [Double]) -> Bool {
    guard rect[2] >= 1, rect[3] >= 1 else { return false }
    return rect[0] < window[0] + window[2] && rect[0] + rect[2] > window[0]
        && rect[1] < window[1] + window[3] && rect[1] + rect[3] > window[1]
}
