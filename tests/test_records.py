

def test_action_repr_matches_mind2web_format() -> None:
    from grounder.records import Element, action_repr

    assert action_repr(Element(id=1, role="link", text="Careers", tag="a")) == "[a]  Careers -> CLICK"
    assert action_repr(Element(id=2, role="searchbox", text="Find a location"), "Boston") == \
        "[searchbox]  Find a location -> TYPE: Boston"
