

def test_shares_words_and_overlap_split():
    from grounder.eval import evaluate, shares_words
    from grounder.ranking import Ranked
    from grounder.records import Element, Example, Page

    page = Page("p", "u", "s", "t", [1, 1], [1, 1], [
        Element(id=1, role="button", text="Sign in", context="account"),
        Element(id=2, role="link", text="Pricing"),
    ])
    lexical = Example("p", "plain", "do", "click sign in", [], 1, [1])
    semantic = Example("p", "paraphrase", "do", "log into my profile", [], 1, [1])
    assert shares_words(page, lexical) and not shares_words(page, semantic)
    ranks = [Ranked([1, 2], [0.9, 0.1], 0.0), Ranked([2, 1], [0.6, 0.4], 0.0)]
    m = evaluate([lexical, semantic], ranks, [True, False])
    assert m["by_overlap"]["shared_words"]["top1"] == 1.0 and m["by_overlap"]["no_shared_words"]["top1"] == 0.0
