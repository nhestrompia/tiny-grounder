() => {
  const MAX_ELEMENTS = 3000;
  const MAX_TEXT = 80;
  const MAX_CONTEXT = 120;
  const MAX_HREF = 200;
  const MAX_HEADING = 40;
  const MAX_ROW_TEXT = 300;
  const MAX_FRAME_DEPTH = 3;
  const SHOW_ELEMENT = 1;

  const INTERACTIVE_ROLES = [
    "button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "combobox", "textbox", "searchbox", "slider", "spinbutton", "treeitem",
  ];
  const CANDIDATE_SELECTOR = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary", "[onclick]",
    "[contenteditable]:not([contenteditable=false])",
    ...INTERACTIVE_ROLES.map((r) => `[role="${r}"]`),
  ].join(",");
  const ROLE_ALIAS = {
    menuitemcheckbox: "menuitem", menuitemradio: "menuitem", spinbutton: "textbox",
    treeitem: "option", img: "image",
  };
  const VOCAB = new Set([
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "switch", "combobox",
    "option", "tab", "menuitem", "slider", "image", "text", "other",
  ]);
  const BUTTON_INPUT_TYPES = new Set(["button", "submit", "reset", "image", "file", "color"]);
  const ROW_SELECTOR = "li,tr,article,[role=listitem],[role=row]";
  const SECTION_SELECTOR =
    "section,form,main,nav,aside,header,footer,[role=region],dialog";
  const HEADING_SELECTOR = "h1,h2,h3,h4,h5,h6,[role=heading]";
  const SKIP_LABEL_TAGS = new Set(["SELECT", "TEXTAREA", "SCRIPT", "STYLE"]);

  const vw = innerWidth;
  const vh = innerHeight;
  const sx = scrollX;
  const sy = scrollY;

  const flat = (s) => (s || "").replace(/\s+/g, " ").trim();
  // Cut to n UTF-16 units without leaving half of an emoji (a lone high surrogate) at the end.
  const cut = (s, n) => s.slice(0, n).replace(/[\uD800-\uDBFF]$/, "");
  const clip = (s, n) => cut(flat(s), n).trim();
  const round = (v, digits) => Math.round(v * 10 ** digits) / 10 ** digits;
  const parentOf = (n) => n.parentElement || n.getRootNode().host || null;
  const cursorOf = (el) => el.ownerDocument.defaultView.getComputedStyle(el).cursor;

  // ---- visibility ----------------------------------------------------------------------

  function isVisible(el) {
    if (typeof el.checkVisibility === "function") {
      return el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
    }
    const win = el.ownerDocument.defaultView;
    if (win.getComputedStyle(el).visibility !== "visible") return false;
    for (let n = el; n; n = parentOf(n)) {
      const cs = win.getComputedStyle(n);
      if (cs.display === "none" || parseFloat(cs.opacity) === 0) return false;
    }
    return true;
  }

  function labelTarget(hit, el) {
    const label = hit.closest && hit.closest("label");
    return !!label && label.control === el;
  }

  function hitsElement(el, x, y) {
    const hit = el.getRootNode().elementFromPoint(x, y);
    if (!hit) return false;
    return hit === el || el.contains(hit) || hit.contains(el) || labelTarget(hit, el);
  }

  // ---- candidates ----------------------------------------------------------------------

  // 'semantic' for explicit interactive markup, 'cursor' for pointer-only leaves, else null.
  function candidateKind(el) {
    if (el.matches(CANDIDATE_SELECTOR)) return "semantic";
    const tabindex = el.getAttribute("tabindex");
    if (tabindex !== null && parseInt(tabindex, 10) >= 0) return "semantic";
    if (isPointerLeaf(el)) return "cursor";
    return null;
  }

  function isPointerLeaf(el) {
    if (el.childElementCount > 2 || el.matches("html,body,svg,svg *")) return false;
    const parent = parentOf(el);
    if (!parent || cursorOf(el) !== "pointer" || cursorOf(parent) === "pointer") return false;
    return !el.querySelector(CANDIDATE_SELECTOR);
  }

  function hrefOf(el) {
    if (el.tagName !== "A" || !el.hasAttribute("href")) return "";
    let url;
    try {
      url = new URL(el.href);
    } catch (e) {
      return clip(el.getAttribute("href"), MAX_HREF);
    }
    const sameOrigin =
      url.origin === location.origin || (url.protocol === "file:" && location.protocol === "file:");
    return (sameOrigin ? url.pathname + url.search + url.hash : url.href).slice(0, MAX_HREF);
  }

  // Returns a measured entry, or null when the element is hidden, empty or covered.
  function measure(el, kind, ctx) {
    if (!isVisible(el)) return null;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return null;
    const x = r.left + ctx.ox;
    const y = r.top + ctx.oy;
    if (x + r.width + sx <= 0 || y + r.height + sy <= 0) return null; // off-screen skip links
    const win = el.ownerDocument.defaultView;
    const inFrame = r.right > 0 && r.bottom > 0 && r.left < win.innerWidth && r.top < win.innerHeight;
    const inViewport = inFrame && x + r.width > 0 && y + r.height > 0 && x < vw && y < vh;
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const centreVisible =
      cx >= 0 && cy >= 0 && cx < win.innerWidth && cy < win.innerHeight &&
      cx + ctx.ox >= 0 && cy + ctx.oy >= 0 && cx + ctx.ox < vw && cy + ctx.oy < vh;
    if (centreVisible && !hitsElement(el, cx, cy)) return null;
    return {
      el, kind, href: hrefOf(el), inViewport,
      abs: [x + sx, y + sy, r.width, r.height],
    };
  }

  function collect(root, ctx, entries) {
    const doc = root.ownerDocument || root;
    const walker = doc.createTreeWalker(root, SHOW_ELEMENT);
    for (let el = walker.nextNode(); el && entries.length < MAX_ELEMENTS; el = walker.nextNode()) {
      try {
        const kind = candidateKind(el);
        const entry = kind && measure(el, kind, ctx);
        if (entry) entries.push(entry);
        if (el.shadowRoot) collect(el.shadowRoot, ctx, entries);
        if (el.tagName === "IFRAME" || el.tagName === "FRAME") enterFrame(el, ctx, entries);
      } catch (e) {
        // A single odd element must never abort the extraction.
      }
    }
  }

  function enterFrame(frame, ctx, entries) {
    if (ctx.depth >= MAX_FRAME_DEPTH || !isVisible(frame)) return;
    let doc;
    try {
      doc = frame.contentDocument; // null (or throws) for cross-origin frames
    } catch (e) {
      return;
    }
    if (!doc || !doc.documentElement) return;
    const r = frame.getBoundingClientRect();
    const child = {
      ox: ctx.ox + r.left + frame.clientLeft,
      oy: ctx.oy + r.top + frame.clientTop,
      depth: ctx.depth + 1,
    };
    collect(doc, child, entries);
  }

  // Keep the outermost interactive element unless the inner one links somewhere else.
  function dropNested(entries) {
    const kept = new Map();
    for (const entry of entries) {
      let outer = null;
      for (let n = parentOf(entry.el); n && !outer; n = parentOf(n)) outer = kept.get(n) || null;
      if (outer && !(entry.href && entry.href !== outer.href)) continue;
      kept.set(entry.el, entry);
    }
    return [...kept.values()];
  }

  // ---- role ----------------------------------------------------------------------------

  function roleOf(el) {
    const tag = el.tagName.toLowerCase();
    const explicit = flat(el.getAttribute("role")).split(" ")[0].toLowerCase();
    if (explicit && explicit !== "presentation" && explicit !== "none") {
      if (explicit === "search" && tag === "input") return "searchbox";
      const role = ROLE_ALIAS[explicit] || explicit;
      return VOCAB.has(role) ? role : "other";
    }
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "img") return "image";
    if (tag === "input") return inputRole(el.type);
    if (el.isContentEditable) return "textbox";
    return "other";
  }

  function inputRole(type) {
    if (BUTTON_INPUT_TYPES.has(type)) return "button";
    if (type === "checkbox" || type === "radio") return type;
    if (type === "range") return "slider";
    if (type === "search") return "searchbox";
    return "textbox";
  }

  // ---- text ----------------------------------------------------------------------------

  const isFormField = (el) => ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName);

  function visibleText(el) {
    if (isFormField(el)) return "";
    return flat(el.innerText !== undefined ? el.innerText : el.textContent);
  }

  // Label text without the control itself (a wrapping label contains the control's options).
  function labelOwnText(label, control) {
    let out = "";
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (child === control) continue;
        if (child.nodeType === 3) out += child.data;
        else if (child.nodeType === 1 && !SKIP_LABEL_TAGS.has(child.tagName)) {
          out += " ";
          walk(child);
        }
      }
    };
    walk(label);
    return flat(out);
  }

  function labelTextOf(el) {
    if (!isFormField(el) || BUTTON_INPUT_TYPES.has(el.type) || !el.labels) return "";
    return flat([...el.labels].map((label) => labelOwnText(label, el)).join(" "));
  }

  function labelledByText(el) {
    const ids = flat(el.getAttribute("aria-labelledby")).split(" ").filter(Boolean);
    const root = el.getRootNode();
    return flat(
      ids.map((id) => {
        const ref = root.getElementById ? root.getElementById(id) : null;
        if (!ref) return "";
        return ref.getAttribute("aria-label") || ref.innerText || ref.textContent;
      }).join(" "),
    );
  }

  // Icon-only links and buttons: alt/title of a descendant image or svg, or an svg <title>.
  function iconText(el) {
    const icon = el.querySelector("img[alt]:not([alt='']),[title]:not([title='']),svg > title");
    if (!icon) return "";
    return flat(icon.getAttribute("alt") || icon.getAttribute("title") || icon.textContent);
  }

  function buttonValue(el) {
    if (el.tagName !== "INPUT" || !["button", "submit", "reset"].includes(el.type)) return "";
    return flat(el.getAttribute("value")) || { submit: "Submit", reset: "Reset" }[el.type] || "";
  }

  function selectedOptionText(el) {
    return el.tagName === "SELECT" && el.selectedOptions[0] ? flat(el.selectedOptions[0].text) : "";
  }

  // Returns {text, own}: the label shown to the model and the element's own rendered text.
  function textOf(el) {
    const own = visibleText(el);
    const candidates = [
      flat(el.getAttribute("aria-label")),
      labelledByText(el),
      labelTextOf(el),
      flat(el.getAttribute("alt")),
      flat(el.getAttribute("placeholder")),
      buttonValue(el),
      own,
      // Tooltips are a fallback: they are often longer than, and differ from, the visible label.
      flat(el.getAttribute("title")),
      iconText(el),
      selectedOptionText(el),
    ];
    return { text: clip(candidates.find(Boolean) || "", MAX_TEXT), own };
  }

  // ---- context -------------------------------------------------------------------------

  const rowTextCache = new WeakMap();

  function rowTextOf(node) {
    if (!rowTextCache.has(node)) rowTextCache.set(node, flat(node.innerText));
    return rowTextCache.get(node);
  }

  function rowAncestor(el, own) {
    let tooBig = false;
    for (let n = parentOf(el); n && n.tagName !== "BODY" && n.tagName !== "HTML"; n = parentOf(n)) {
      if (n.matches(ROW_SELECTOR)) return n;
      if (n.tagName !== "DIV" || tooBig) continue;
      if (n.textContent.length > MAX_ROW_TEXT * 4) {
        tooBig = true;
        continue;
      }
      const text = rowTextOf(n);
      if (text.length > MAX_ROW_TEXT) tooBig = true;
      else if (text.length > own.length && text !== own) return n;
    }
    return null;
  }

  const headingCache = new WeakMap();

  // Headings that belong directly to `scope`, not to a section nested inside it.
  function headingsIn(scope, fallback) {
    if (!headingCache.has(scope)) {
      const all = [...scope.querySelectorAll(HEADING_SELECTOR)];
      headingCache.set(scope, all.filter((h) => (h.closest(SECTION_SELECTOR) || fallback) === scope));
    }
    return headingCache.get(scope);
  }

  function precedes(heading, el) {
    return !!(heading.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
  }

  // Nearest heading before `el` inside its closest section/form/landmark.
  function headingBefore(el) {
    const root = el.getRootNode();
    const fallback = root.body || root;
    const scope = el.closest(SECTION_SELECTOR) || fallback;
    const headings = headingsIn(scope, fallback);
    let lo = 0;
    let hi = headings.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (precedes(headings[mid], el)) lo = mid + 1;
      else hi = mid;
    }
    for (let i = lo - 1; i >= 0; i--) {
      const h = headings[i];
      if (h.contains(el)) continue;
      const text = flat(h.getAttribute("aria-label") || h.innerText);
      if (text) return text;
    }
    return "";
  }

  function legendOf(el) {
    for (let n = parentOf(el); n; n = parentOf(n)) {
      if (n.tagName === "FIELDSET") {
        const legend = n.querySelector(":scope > legend");
        return legend ? flat(legend.textContent) : "";
      }
    }
    return "";
  }

  // Remove the element's own text; trim separators (| · •) it leaves dangling at the ends.
  function without(text, own) {
    const rest = own ? text.split(own).join(" ") : text;
    return flat(rest.replace(/^[\s|·•]+|[\s|·•]+$/g, ""));
  }

  function joinContext(primary, secondary) {
    const second = clip(secondary, MAX_HEADING);
    const budget = MAX_CONTEXT - (second ? second.length + 3 : 0);
    return [clip(primary, budget), second].filter(Boolean).join(" · ");
  }

  function contextOf(el, text, own) {
    if (isFormField(el)) {
      const label = labelTextOf(el);
      const fieldContext = joinContext(label === text ? "" : label, legendOf(el));
      if (fieldContext) return fieldContext;
    }
    const row = rowAncestor(el, own);
    let rowText = row ? without(without(rowTextOf(row), own), text) : "";
    if (rowText === text) rowText = "";
    const heading = headingBefore(el);
    const redundant = heading === text || (heading && rowText.includes(heading));
    return joinContext(rowText, redundant ? "" : heading);
  }

  // ---- assemble ------------------------------------------------------------------------

  function describe(entry) {
    const { el, kind, href, inViewport, abs } = entry;
    const { text, own } = textOf(el);
    if (kind === "cursor" && !text) return null;
    return {
      role: roleOf(el),
      text,
      context: contextOf(el, text, own),
      tag: el.tagName.toLowerCase(),
      box: [abs[0] / vw, abs[1] / vh, abs[2] / vw, abs[3] / vh].map((v) => round(v, 4)),
      abs_box: abs.map((v) => round(v, 1)),
      in_viewport: inViewport,
      enabled: !el.matches(":disabled") && !el.closest('[aria-disabled="true"]'),
      conf: 1.0,
      source: "dom",
      href,
    };
  }

  const entries = [];
  collect(document, { ox: 0, oy: 0, depth: 0 }, entries);
  const elements = [];
  for (const entry of dropNested(entries)) {
    try {
      const described = describe(entry);
      if (described) elements.push({ id: elements.length, ...described });
    } catch (e) {
      // Skip elements whose text or context cannot be computed.
    }
  }
  const root = document.documentElement;
  return {
    viewport: [vw, vh],
    doc_size: [Math.max(root.scrollWidth, vw), Math.max(root.scrollHeight, vh)],
    title: flat(document.title),
    elements,
  };
}
