"""Pinned competitor identities and prompt descriptions. No binary is downloaded at run time."""
from pathlib import Path

PW_VERSION = "0.0.84"
AB_VERSION = "0.41.2"
BIN_DIR = Path(__file__).resolve().parents[1] / ".cache/competitors/node_modules/.bin"
COMPETITOR_CONFIGS = ("pw-mcp", "pw-mcp-compact", "agent-browser", "pw-mcp+observe")
COMPETITOR_HINTS = {
    "pw-mcp": "Call browser_snapshot() to view the page and use its refs as browser_click(target). Actions return a page summary that may include a snapshot or a link to it. Call browser_snapshot() for the current view.",
    "pw-mcp-compact": "Call browser_snapshot(depth=3) or request a target subtree for a compact view. Use browser_find(text) to locate elements and their refs, then pass a ref as the target of Playwright actions. Actions return no automatic view.",
    "agent-browser": "Call snapshot() for interactive elements with refs, using agent-browser's snapshot -i -c mode. Use click(ref) or fill(ref, text) with an @ref, open(url) to navigate, back() to return, and get_text(ref) to read text.",
    "pw-mcp+observe": "Call observe() for a goal digest, find(query) for controls, read(query) for facts, and inspect(id) for details. Pass ids from observe/find as the target of Playwright actions. Actions return a fresh goal digest.",
}

# Prompt names only: the actual Playwright tools/list is fetched unchanged from the binary.
def competitor_tools(config):
    if config == "agent-browser":
        return ("snapshot", "click", "fill", "select", "press", "open", "back", "get_text", "done")
    names = ("browser_snapshot", "browser_find", "browser_click", "browser_type", "browser_navigate", "browser_navigate_back")
    if config == "pw-mcp+observe":
        return names[2:] + ("observe", "find", "inspect", "read", "done")
    return names + ("done",)


def pw_args(config, endpoint):
    return ["--cdp-endpoint", endpoint, "--timeout-settle", "0"] + ([] if config == "pw-mcp" else ["--snapshot-mode", "none"])

# Complete inventory from the pinned Playwright binary, including optional capabilities.
# False means no owner load/settle wait. Arbitrary code and unknown tools are conservative.
PW_PAGE_CHANGING = {
    'browser_annotate': True,
    'browser_check': True,
    'browser_click': True,
    'browser_close': True,
    'browser_console_clear': False,
    'browser_console_messages': False,
    'browser_cookie_clear': True,
    'browser_cookie_delete': True,
    'browser_cookie_get': False,
    'browser_cookie_list': False,
    'browser_cookie_set': True,
    'browser_drag': True,
    'browser_drop': True,
    'browser_emulate_media': True,
    'browser_evaluate': True,
    'browser_file_upload': True,
    'browser_fill_form': True,
    'browser_find': False,
    'browser_generate_locator': False,
    'browser_get_config': False,
    'browser_handle_dialog': True,
    'browser_hide_highlight': True,
    'browser_highlight': True,
    'browser_hover': True,
    'browser_keydown': True,
    'browser_keyup': True,
    'browser_localstorage_clear': True,
    'browser_localstorage_delete': True,
    'browser_localstorage_get': False,
    'browser_localstorage_list': False,
    'browser_localstorage_set': True,
    'browser_mouse_click_xy': True,
    'browser_mouse_down': True,
    'browser_mouse_drag_xy': True,
    'browser_mouse_move_xy': True,
    'browser_mouse_up': True,
    'browser_mouse_wheel': True,
    'browser_navigate': True,
    'browser_navigate_back': True,
    'browser_navigate_forward': True,
    'browser_network_clear': False,
    'browser_network_request': False,
    'browser_network_requests': False,
    'browser_network_state_set': True,
    'browser_pdf_save': False,
    'browser_press_key': True,
    'browser_press_sequentially': True,
    'browser_reload': True,
    'browser_resize': True,
    'browser_resume': True,
    'browser_route': False,
    'browser_route_list': False,
    'browser_run_code_unsafe': True,
    'browser_select_option': True,
    'browser_sessionstorage_clear': True,
    'browser_sessionstorage_delete': True,
    'browser_sessionstorage_get': False,
    'browser_sessionstorage_list': False,
    'browser_sessionstorage_set': True,
    'browser_set_storage_state': True,
    'browser_snapshot': False,
    'browser_start_recording': False,
    'browser_start_tracing': False,
    'browser_start_video': False,
    'browser_stop_recording': False,
    'browser_stop_tracing': False,
    'browser_stop_video': False,
    'browser_storage_state': False,
    'browser_tabs': True,
    'browser_take_screenshot': False,
    'browser_type': True,
    'browser_uncheck': True,
    'browser_unroute': False,
    'browser_verify_element_visible': False,
    'browser_verify_list_visible': False,
    'browser_verify_text_visible': False,
    'browser_verify_value': False,
    'browser_video_chapter': False,
    'browser_video_hide_actions': False,
    'browser_video_show_actions': False,
    'browser_wait_for': False,
    'browser_webmcp_call': True,
    'browser_webmcp_list': False,
}

AB_PAGE_CHANGING = {"snapshot": False, "get_text": False, "click": True, "fill": True, "select": True,
                    "press": True, "open": True, "back": True}

def page_changing(name, args):
    if name == "browser_tabs":
        return args.get("action") != "list"
    return PW_PAGE_CHANGING.get(name, AB_PAGE_CHANGING.get(name, True))
