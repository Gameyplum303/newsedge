import pytest

from engine import sanitize, store


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,<script>x</script>", "file:///C:/x",
                                 "//evil.com/a", "https://", "", None])
def test_only_plain_web_links_are_kept(url):
    assert sanitize.safe_link(url) == ""


def test_web_links_pass_through():
    assert sanitize.safe_link(" https://www.cnbc.com/a?b=1 ") == "https://www.cnbc.com/a?b=1"


@pytest.mark.parametrize("ticker", ["AAPL", "BRK-B", "BF.B", "GOOGL"])
def test_valid_tickers(ticker):
    assert sanitize.valid_ticker(ticker)


@pytest.mark.parametrize("ticker", ["../etc", "AAPL&x=1", "aapl", "", "A B", "AAPL\n", 7, "TOOLONGTICKER"])
def test_invalid_tickers(ticker):
    assert not sanitize.valid_ticker(ticker)


def test_hand_edited_watchlist_is_cleaned():
    raw = {"AAPL": ["Apple", 3, ""], "../x": ["Bad"], "MSFT": "Microsoft", "NVDA": ["Nvidia"]}
    assert store._clean_watchlist(raw) == {"AAPL": ["Apple"], "NVDA": ["Nvidia"]}
    assert store._clean_watchlist(["AAPL"]) == {}


def test_saving_an_invalid_ticker_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "WATCHLIST_PATH", str(tmp_path / "w.json"))
    with pytest.raises(ValueError):
        store.save_watchlist({"AAPL": ["Apple"], "A;B": []})
    assert not (tmp_path / "w.json").exists()


def test_job_updates_only_touch_known_columns():
    with pytest.raises(ValueError):
        store.update_job(1, **{"status = 'done' --": "x"})


def test_feed_html_becomes_plain_text():
    assert sanitize.plain_text("<b>Apple</b> &amp; <script>x</script>\n Nvidia") == "Apple & x Nvidia"


def test_headlines_cannot_inject_markdown():
    text = sanitize.markdown_literal("Click [here](https://evil.example) ![x](https://t.example/p.png) **now**")
    assert "](" not in text.replace("\\](", "") and "![" not in text.replace("\\!\\[", "")
    assert text.startswith("Click \\[here\\]\\(")
