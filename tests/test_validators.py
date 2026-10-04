"""Simple checks, no extra packages needed.  Run:  python -m tests.test_validators"""
from app.validators import (strip_urls, has_url, fit_description, fit_title, make_goal, CATEGORY_ORDER)


def test_urls_removed():
    assert not has_url(strip_urls("Visit https://a.com/help or www.b.org or [x](http://c.com) or d.com/support"))


def test_description_rule():
    for raw in ["Allows you to switch navigation type", "It will let you choose navigation type quickly right now", "fix"]:
        d = fit_description(raw)
        assert d.startswith("It will"), d
        assert 5 <= len(d.split()) <= 7, d


def test_title_rule():
    for raw in ["Swipe navigation settings fix now", "Battery"]:
        assert 2 <= len(fit_title(raw).split()) <= 3


def test_goal_format():
    g = make_goal("Screen Flicker troubleshooting")
    assert g == "Follow these steps to perform this Screen Flicker Troubleshooting", g


def test_ordering():
    assert CATEGORY_ORDER["auto"] < CATEGORY_ORDER["manual"] < CATEGORY_ORDER["critical"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
