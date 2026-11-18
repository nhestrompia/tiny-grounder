"""English-only page filter: the score, `is_english` with `lang`, and the build / teacher integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from grounder.data import build
from grounder.data.language import (
    DEFAULT_MIN_ENGLISH, LANG_EN_MIN_SCORE, english_pages, english_score, is_english, primary_subtag,
)
from grounder.data.pages import load_examples, load_pages, write_examples, write_page
from grounder.data.teacher import TeacherConfig, generate, pick_pages
from grounder.records import Element, Example, Page

LABELS = {
    "en": ["Home", "About us", "Contact", "Sign in", "Search", "Cart", "Shop all", "New arrivals", "Help center",
           "Privacy policy", "Terms of service", "Subscribe to our newsletter"],
    "ko": ["홈", "회사 소개", "문의하기", "로그인", "검색", "장바구니", "전체 상품", "신상품", "고객센터",
           "개인정보 처리방침", "이용약관", "뉴스레터 구독"],
    "zh": ["首页", "关于我们", "联系我们", "登录", "搜索", "购物车", "全部商品", "新品上市", "帮助中心",
           "隐私政策", "服务条款", "订阅我们的通讯"],
    "ru": ["Главная", "О компании", "Контакты", "Войти", "Поиск", "Корзина", "Все товары", "Новинки", "Помощь",
           "Политика конфиденциальности", "Условия использования", "Подписаться на рассылку"],
    "de": ["Startseite", "Über uns", "Kontakt", "Anmelden", "Suche", "Warenkorb", "Alle Produkte", "Neuheiten",
           "Hilfe", "Datenschutzerklärung", "Impressum", "Newsletter abonnieren"],
    "es": ["Inicio", "Sobre nosotros", "Contacto", "Iniciar sesión", "Buscar", "Carrito", "Todos los productos",
           "Novedades", "Ayuda", "Política de privacidad", "Términos y condiciones", "Suscríbete al boletín"],
    "fr": ["Accueil", "À propos", "Contactez-nous", "Se connecter", "Rechercher", "Panier", "Tous les produits",
           "Nouveautés", "Aide", "Politique de confidentialité", "Mentions légales", "Abonnez-vous à la newsletter"],
}
BRANDS = ["Zalando", "Spotify", "Xiaomi", "Pinterest", "Sennheiser HD 600", "Home", "Sign in", "Cart", "Search",
          "Zorbix Pro", "Shop", "Help"]
MIXED = ["Home", "Search", "Cart", "Kontakt", "Warenkorb", "Anmelden", "Impressum", "Hilfe"]
NON_ENGLISH = [lang for lang in LABELS if lang != "en"]


def page_of(texts: list[str], page_id: str = "p", title: str = "", lang: str = "") -> Page:
    elements = [Element(id=i, role="link", text=t, href=f"/{i}", box=[0.1, 0.05 * i, 0.1, 0.04],
                        abs_box=[128, 40.0 * i, 128, 32]) for i, t in enumerate(texts)]
    return Page(page_id, f"https://{page_id}.test/", f"{page_id}.test", title, [1280, 800], [1280, 4000], elements, lang)


def test_english_ui_scores_high():
    assert english_score(page_of(LABELS["en"], title="Acme Shop - Home")) > 0.9
    assert is_english(page_of(LABELS["en"]))


@pytest.mark.parametrize("lang", NON_ENGLISH)
def test_other_languages_are_rejected(lang):
    page = page_of(LABELS[lang])
    assert english_score(page) < DEFAULT_MIN_ENGLISH - 0.2
    assert not is_english(page)


def test_brand_names_do_not_count_against_a_page():
    assert is_english(page_of(BRANDS, title="Zorbix"))
    assert is_english(page_of(["Acme", "Zorbix"]))  # nothing recognisable either way


def test_short_pages():
    assert is_english(page_of(["Home", "Contact"]))
    assert not is_english(page_of(["Kontakt", "Impressum", "Anmelden"]))
    assert not is_english(page_of(["Inicio", "Contacto"]))
    assert not is_english(page_of(["ログイン", "検索"]))


def test_empty_page_has_nothing_against_it():
    assert english_score(page_of([])) == 1.0


def test_score_is_in_unit_range_and_deterministic():
    for texts in LABELS.values():
        page = page_of(texts, title="Title")
        assert 0.0 <= english_score(page) <= 1.0
        assert english_score(page) == english_score(page)


def test_contexts_count_as_words():
    page = page_of(["Zorbix", "Qlip"])
    page.elements[0].context = "Sign in to your account to view all orders"
    assert english_score(page) == 1.0
    page.elements[0].context = "Anmelden und alle Bestellungen ansehen Kontakt Hilfe Suche"
    assert english_score(page) < DEFAULT_MIN_ENGLISH


def test_language_picker_does_not_make_an_english_page_foreign():
    picker = ["English", "Français", "Deutsch", "日本語", "한국어", "Русский", "中文", "العربية"]
    assert is_english(page_of(LABELS["en"] + picker))


def test_declared_non_english_lang_overrides_the_score():
    assert not is_english(page_of(LABELS["en"], lang="de"))
    assert not is_english(page_of(LABELS["en"], lang="fr-CA"))
    assert not is_english(page_of(LABELS["en"], lang="zh_CN"))


def test_declared_english_lang_is_not_enough_on_its_own():
    assert is_english(page_of(LABELS["en"], lang="en-GB"))
    assert is_english(page_of(LABELS["en"], lang="EN"))
    assert not is_english(page_of(LABELS["ko"], lang="en"))
    assert not is_english(page_of(LABELS["de"], lang="en"))


def test_declared_english_lang_lowers_the_bar_for_a_mixed_page():
    score = english_score(page_of(MIXED))
    assert LANG_EN_MIN_SCORE <= score < DEFAULT_MIN_ENGLISH
    assert not is_english(page_of(MIXED))
    assert is_english(page_of(MIXED, lang="en"))


def test_primary_subtag():
    assert [primary_subtag(x) for x in ("en", "EN-us", "pt_BR", " de ")] == ["en", "en", "pt", "de"]


def test_english_pages_filter_and_off_switch():
    pages = [page_of(LABELS["en"], "a"), page_of(LABELS["ko"], "b"), page_of(LABELS["en"], "c", lang="ru")]
    assert [p.page_id for p in english_pages(pages)] == ["a"]
    assert [p.page_id for p in english_pages(pages, 0)] == ["a", "b", "c"]


def test_old_page_files_load_without_lang():
    d = page_of(LABELS["en"]).to_dict()
    del d["lang"]
    assert Page.from_dict(d).lang == ""
    assert Page.from_dict({**d, "lang": "de-AT"}).lang == "de-AT"


# --- integration -----------------------------------------------------------------------------


def write_crawl(root: Path) -> dict[str, Page]:
    """A train split with two English pages and a Korean, a German and a `lang="de"` page."""
    pages = {
        "en1": page_of(LABELS["en"], "en1", "Acme Shop"),
        "en2": page_of(BRANDS, "en2", "Zorbix"),
        "ko1": page_of(LABELS["ko"], "ko1"),
        "de1": page_of(LABELS["de"], "de1"),
        "de2": page_of(LABELS["en"], "de2", lang="de"),
    }
    for page in pages.values():
        write_page(page, root / "train")
    return pages


def test_build_skips_non_english_pages(tmp_path, capsys):
    write_crawl(tmp_path / "pages")
    llm = [Example(pid, "paraphrase", "do", "go to the start page", [], 0, [0]) for pid in ("en1", "ko1", "de2")]
    write_examples(llm, tmp_path / "llm.jsonl")
    argv = ["--pages", str(tmp_path / "pages"), "--out", str(tmp_path / "out"), "--llm-examples", str(tmp_path / "llm.jsonl")]
    assert build.main(argv) == 0
    assert "train: 2 English pages, 3 dropped" in capsys.readouterr().out
    used = {ex.page_id for ex in load_examples(tmp_path / "out" / "train.jsonl")}
    assert used == {"en1", "en2"}

    assert build.main([*argv, "--min-english", "0"]) == 0
    used_all = {ex.page_id for ex in load_examples(tmp_path / "out" / "train.jsonl")}
    assert {"ko1", "de1", "de2"} <= used_all


def test_teacher_only_sees_english_pages(tmp_path):
    write_crawl(tmp_path / "pages")
    picked = pick_pages(tmp_path / "pages", None, 0, DEFAULT_MIN_ENGLISH)
    assert [p.page_id for p in picked] == ["en1", "en2"]
    assert len(pick_pages(tmp_path / "pages", None, 0, 0)) == 5
    assert [p.page_id for p in pick_pages(tmp_path / "pages", 1, 0, DEFAULT_MIN_ENGLISH)] in (["en1"], ["en2"])

    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"items": []})}}]})

    async def run() -> None:
        cfg = TeacherConfig(model="m", cache_dir=tmp_path / "cache", backoff_s=0)
        async with httpx.AsyncClient(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler)) as http:
            await generate(picked, http, cfg)

    asyncio.run(run())
    assert prompts and all("Acme Shop" in p or "Zorbix" in p for p in prompts)
    assert not any("로그인" in p or "Anmelden" in p for p in prompts)


def test_loaded_pages_keep_lang(tmp_path):
    write_crawl(tmp_path)
    loaded = load_pages(tmp_path)
    assert loaded["de2"].lang == "de" and loaded["en1"].lang == ""
