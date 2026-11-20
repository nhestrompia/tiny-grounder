from tools.leakage_audit import (
    MIN_TEXTS, TrainIndex, app_forms, audit_source, jaccard, normalize_url, page_info, site_domain_key,
)


def page(url: str, texts: list[str], page_id: str = "p") -> dict:
    return {"page_id": page_id, "url": url, "site": url.split("://")[1].split("/")[0],
            "elements": [{"id": i, "text": t} for i, t in enumerate(texts)]}


def texts(prefix: str, n: int = 20) -> list[str]:
    return [f"{prefix} item {i}" for i in range(n)]


def test_normalize_url() -> None:
    assert normalize_url("https://www.Example.com/a/b/?b=2&a=1#frag") == "example.com/a/b?a=1&b=2"
    assert normalize_url("http://example.com/") == "example.com"
    assert normalize_url("https://example.com") == normalize_url("http://www.example.com/")


def test_page_info_kinds() -> None:
    w = page_info("web", page("https://shop.example.co.uk/x", ["a1", "b2"]))
    assert w.domain == "example.co.uk" and w.app == ""
    d = page_info("deskforge", page("ax://deskforge/gnome-calculator", ["a1"]))
    assert d.app == "gnome-calculator" and d.domain == ""
    g = page_info("guirilla_test", page("ax://com.apple.Notes", ["a1"]))
    assert g.app == "com.apple.Notes"
    s = page_info("s2ax", page("ax://s2ax-003ee21c", ["a1"]))
    assert s.app == ""
    m = page_info("mind2web", page("mind2web://new.mta.info", ["a1"]))
    assert m.domain == "mta.info" and site_domain_key("kohls") == "kohls"


def test_app_forms() -> None:
    assert "notes" in app_forms("com.apple.Notes")
    assert app_forms("notes") & app_forms("com.apple.Notes")
    assert "calculator" in app_forms("gnome-calculator")
    assert not app_forms("ata_")  # too short to match on


def test_domain_url_and_app_overlap() -> None:
    idx = TrainIndex()
    idx.add(page_info("web", page("https://www.shop.com/a?x=1", texts("shop"))))
    idx.add(page_info("ax", page("ax://notes", texts("notes"))))
    idx.add(page_info("mind2web", page("mind2web://kohls", texts("kohls"))))
    ev = [
        page_info("web_test", page("https://shop.com/other", texts("zzz"), page_id="1")),  # same domain
        page_info("web_test", page("http://shop.com/a/?x=1", texts("yyy"), page_id="2")),  # same URL
        page_info("web_test", page("https://other.org/", texts("qqq"), page_id="3")),
    ]
    r = audit_source("web_test", ev, idx)
    assert (r.pages, r.domain_pages, r.url_pages) == (3, 2, 1)
    assert set(r.overlap_domains) == {"shop.com"}
    apps = audit_source("guirilla_test", [
        page_info("guirilla_test", page("ax://com.apple.Notes", texts("n"))),
        page_info("guirilla_test", page("ax://com.other.Thing", texts("t"))),
    ], idx)
    assert apps.app_pages == 1 and "com.apple.Notes" in apps.overlap_apps
    m2w = audit_source("mind2web_test_task", [page_info("mind2web_test_task", page("mind2web://kohls", texts("k")))], idx)
    assert m2w.domain_pages == 1


def test_mind2web_site_matches_web_sld() -> None:
    idx = TrainIndex()
    idx.add(page_info("web", page("https://www.kohls.com/", texts("a"))))
    r = audit_source("mind2web_test_website", [page_info("mind2web_test_website", page("mind2web://kohls", texts("b")))], idx)
    assert r.domain_pages == 1


def test_near_duplicate_detection() -> None:
    base = texts("base", 40)
    idx = TrainIndex()
    idx.add(page_info("web", page("https://a.com/", base)))
    near = base[:-2] + ["new one", "new two"]  # 38 shared / 42 union = 0.905
    far = base[:20] + texts("other", 20)
    ev = [
        page_info("web_test", page("https://b.com/", base, page_id="exact")),
        page_info("web_test", page("https://c.com/", near, page_id="near")),
        page_info("web_test", page("https://d.com/", far, page_id="far")),
        page_info("web_test", page("https://a.com/p", near, page_id="same-domain")),
    ]
    r = audit_source("web_test", ev, idx)
    assert r.near_pages == 3 and r.exact_pages == 1
    assert r.near_cross_pages == 2  # the same-domain copy is a near-dup but not cross-domain
    assert abs(jaccard(frozenset(base), frozenset(near)) - 38 / 42) < 1e-9


def test_small_pages_not_compared() -> None:
    small = ["close", "minimize", "zoom"]
    assert len(small) < MIN_TEXTS
    idx = TrainIndex()
    idx.add(page_info("ax", page("ax://notes", small)))
    r = audit_source("g", [page_info("g", page("ax://other", small))], idx)
    assert r.near_pages == 0 and r.comparable_pages == 0
