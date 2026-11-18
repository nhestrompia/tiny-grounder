"""English-only page filter. Standard library plus `grounder.records` only.

The project is English-only by design, but the crawl also contains pages in other languages. `english_score`
rates a page in [0, 1] and `is_english` applies a threshold and the page's `<html lang>`. The score is the
product of three factors (see `english_score`): script share, English share of the known common words, and
English word coverage. Tokens that are in no word list (brand names, product names) are neither for nor against
a page, so short UI strings and brand-heavy pages are not punished for what the lists do not know.
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable, Iterator

from grounder.records import Page

DEFAULT_MIN_ENGLISH = 0.65
MIN_TOKEN_LETTERS = 3
LANG_EN_MIN_SCORE = 0.35  # a page that declares `lang="en"` still needs this much: many sites ship it untranslated
SCRIPT_PENALTY = 2.0
COVERAGE_PRIOR = 3.0
EXPECTED_ENGLISH_SHARE = 0.1
COVERAGE_MAX_TOKENS = 100  # long pages of names (listings, categories) are not expected to be proportionally English

_TOKEN_RE = re.compile(r"[^\W\d_]+")

ENGLISH_WORDS = frozenset("""
the and for are but not you your all any can had her was
one our out has have his how its may new now old see two
way who did get let put say she too use that this with from
they will would there their what about which when make like time just him
know take into year good some could them than then look only come over
think also back after work first well even want because these give most been
were more here other many much such very where while should home contact sign
log login logout register search menu cart news help privacy terms account shop view
read next previous prev skip main content navigation nav close open save share follow
subscribe download upload submit send cancel continue done apply reset clear filter sort show
hide settings profile preferences accept reject agree policy conditions legal copyright rights reserved company
careers jobs press support faq feedback products product pricing price free buy sale deals
deal offers offer store stores checkout order orders payment shipping delivery returns wishlist bag
basket items item total quantity add remove edit delete update create page pages map
sitemap link links photo photos image images watch weather world politics entertainment food english
language country select choose enter email password username name phone address city state zip
message comment comments reply post posts likes tweet story stories article articles latest popular
trending featured recommended related results result load loading please try again error welcome hello
thanks thank yes no ok started learn courses course academic admissions programs events event
calendar people team leadership board mission values history resources resource documentation docs developers developer
community customers customer investors investor directory categories category archive archives week month day today
tomorrow yesterday hours closed available sold details description specs reviews review rating ratings questions
answers question answer ask tips guide guides tutorial tutorials tools tool library gallery trial
demo request quote quotes call text chat talk meet visit find locations location near
directions men women kids boys girls baby garden beauty fashion clothing shoes accessories electronics
furniture books
""".split())

FOREIGN_WORDS = frozenset("""
der die das den dem des ein eine einen einem einer und oder nicht mit von für fur auf ist sind wird werden auch nach
bei aus über uber zum zur beim dass wie noch nur mehr wir sie ihr ihre sich zurück zuruck startseite impressum
datenschutz kontakt anmelden suche suchen warenkorb hilfe aktuelles nachrichten produkte angebote kundenservice
alle uns jetzt hier presse unternehmen neue einstellungen weitere unsere diese themen informationen kann aktuelle
unter haben durch sprache seite mein karriere ohne bis vor ich hat
los las del una unos unas por con para que como más mas pero sus este esta estos son está estan inicio contacto
buscar ayuda noticias productos servicios ofertas ingresar entrar iniciar sesión sesion carrito tienda nosotros
politica política privacidad términos terminos cuenta todo todos nuestra nuestro aviso accesibilidad datos
información seguridad aceptar rechazar condiciones publicidad nueva nuevo contenido todas sin años día
les des une dans pour avec sur est sont pas qui que vous nous votre vos notre nos plus aux ces ses leurs
accueil recherche rechercher panier aide actualités actualites contactez connexion compte mentions légales
générales générale produits boutique voir tout données politique accessibilité par accepter personnelles tous
sans mes mon partenaires nouvelle refuser savoir toutes choix monde fermer télécharger cette ligne gérer
confidentialité espace protection
uma não nao são sao mais seu sua você voce nós pelo pela início notícias pesquisar cadastrar carrinho ajuda
contato sobre privacidade segurança página conta termos conteúdo serviços informações sem canal aqui fale
conheça plano empresas fazer veja saiba até dados anterior minha busca inicial dos contactos condições
che per sono gli della delle dei degli nel nella alla più piu anche tutti chi siamo accedi cerca carrello
aiuto notizie prodotti offerte contatti registrati accetta informativa servizi dell sui pubblicità nostri
senza dichiarazione sulla tutte sul sito informazioni sicurezza tra tua continua cosa casa noi torna
personalizza alle altri ricerca dati questa chiudi
het een van zijn voor niet met als maar ook dit dat naar meer wij ons onze inloggen zoeken winkelwagen hulp
nieuws producten aanmelden aan vragen alle vacatures bekijk veelgestelde wat lees voorwaarden tot nieuwe
informatie pagina kies zoek werken door hoe onder jouw vind onderzoek weigeren gratis deze volgende mijn jaar
och att det som inte har den ett till från vara hem sök hjälp nyheter kontakta oss här mer bli tjänster fler
följ visa nya eller svenska tillgänglighet din hitta våra nytt jobba inställningar dina när jag min hur
ändra tillbaka efter nästa vill vad frågor
dan yang untuk dengan ini itu pada dari atau tidak anda kami kita berita beranda masuk daftar cari bantuan
layanan kesehatan publik informasi pilihan kontak tautan profil depan pencarian ikuti tentang kebijakan
kegiatan bahasa peraturan lainnya laporan pedoman keuangan terbaik rumah panduan hari
ile bir için ama daha çok gibi ana sayfa giriş ürünler hakkımızda iletişim haberler yardım tüm kişisel yeni
reklam dünya başvuru seçim erişilebilirlik çerez gizlilik hesap hizmetleri ödeme spor yerel kullanım
nie się jest że dla jak lub oraz strona główna wyszukaj pomoc aktualności produkty zaloguj prywatności więcej
polityka informacje ustawienia regulamin reklama nas wszystkie bez usługi przejdź danych szukaj sprawdź już
konto strony pomocy prywatność praca dane czy zobacz firm przez
og ikke med det der som har vil til fra om vores nyheder søg søk hjem logg meny partnere indstillinger præferencer
samtykke detaljer annonceindstillinger forbrugsguides
gebraucht merken anzeigen kostenlos neu melden anmeldung erforderlich verpackung
daftar isi cari kunci lagu
pro jsou nás více nastavení zprávy hledat služby další přihlásit souhlasím podrobné kontakty kariéra
pro hledání zobrazit všechny novinky aktuálně obchodní vše podle nové
nem volt itt süti adatvédelmi szabályzat még már mert így
livrare pentru noi prin utilizare conditii toate anunturi fara produse suport din privind cum vezi frecvente
publicitate contul oferte tale
và của các cho một những không được trong này với khi chúng tôi đăng nhập tìm kiếm trang chủ liên hệ
""".split())
FOREIGN_ONLY = FOREIGN_WORDS - ENGLISH_WORDS


def own_text(page: Page) -> str:
    """The page's own strings: title and element texts."""
    return "\n".join([page.title, *(el.text for el in page.elements)])


def page_text(page: Page) -> str:
    """`own_text` plus the element contexts."""
    return "\n".join([own_text(page), *(el.context for el in page.elements)])


def script_share(text: str) -> float:
    """Share of alphabetic characters that are ASCII letters; 1.0 when there are no letters at all."""
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 1.0


def word_tokens(text: str) -> set[str]:
    """Distinct lower-cased word tokens of at least `MIN_TOKEN_LETTERS` letters.

    Distinct, so a label repeated in a text and in the contexts around it counts once.
    """
    return {t for t in _TOKEN_RE.findall(text.casefold()) if len(t) >= MIN_TOKEN_LETTERS}


def english_score(page: Page) -> float:
    """How English a page looks, in [0, 1]: the product of three factors.

    script:   1 minus `SCRIPT_PENALTY` times the non-ASCII share of the letters of the title and element texts
              (contexts are left out: they repeat a language picker in every neighbour). English UI has almost
              no accented letters, while Czech, Polish or Vietnamese have 5-15%.
    words:    smoothed English share of the tokens that are known words of some language.
    coverage: English words relative to what an English page of this many tokens has; this catches languages
              the foreign list does not know (Uzbek, Finnish, ...). It only bites on pages with many tokens.
    """
    tokens = word_tokens(page_text(page))
    english = len(tokens & ENGLISH_WORDS)
    foreign = len(tokens & FOREIGN_ONLY)
    script = max(0.0, 1.0 - SCRIPT_PENALTY * (1.0 - script_share(own_text(page))))
    words = (english + 1) / (english + foreign + 1)
    expected = EXPECTED_ENGLISH_SHARE * min(len(tokens), COVERAGE_MAX_TOKENS)
    coverage = min(1.0, (english + COVERAGE_PRIOR) / (expected + COVERAGE_PRIOR))
    return script * words * coverage


def primary_subtag(lang: str) -> str:
    return re.split(r"[-_]", lang.strip().casefold())[0]


def is_english(page: Page, threshold: float = DEFAULT_MIN_ENGLISH) -> bool:
    """English by score. A declared `lang` that is not English rules the page out; a declared `en` lowers the bar."""
    score = english_score(page)
    if not page.lang:
        return score >= threshold
    if primary_subtag(page.lang) != "en":
        return False
    return score >= min(threshold, LANG_EN_MIN_SCORE)


def english_pages(pages: Iterable[Page], min_english: float = DEFAULT_MIN_ENGLISH) -> Iterator[Page]:
    """The English pages of `pages`; `min_english` <= 0 turns the filter off."""
    for page in pages:
        if min_english <= 0 or is_english(page, min_english):
            yield page


def add_min_english_argument(parser: argparse.ArgumentParser) -> None:
    """The one `--min-english` option shared by every command that reads crawled pages."""
    parser.add_argument("--min-english", type=float, default=DEFAULT_MIN_ENGLISH,
                        help="drop pages whose english_score is below this (0 keeps every page)")
