"""Ticker eşleştirme regex davranışını doğrula — false positive yakala."""
from ingestion.news_collector import match_tickers


def test_match_single_ticker_by_name():
    assert "THYAO.IS" in match_tickers("Türk Hava Yolları rekor kar açıkladı")


def test_match_by_symbol_uppercase():
    assert "ASELS.IS" in match_tickers("ASELS bugün +%5 yükseldi")


def test_match_multiple_tickers():
    txt = "Garanti BBVA ve Akbank birinci çeyrek karlarını açıkladı"
    out = match_tickers(txt)
    assert "GARAN.IS" in out
    assert "AKBNK.IS" in out


def test_no_match_for_unrelated_text():
    assert match_tickers("Bugün hava çok güzel İstanbul'da") == []


def test_word_boundary_avoids_partial_matches():
    # "AKBNKLI" gibi uzantı varsa match etmemeli (whole word)
    # Not: \b regex word boundary Türkçe karakterlerde de çalışır
    assert "AKBNK.IS" not in match_tickers("Bu sözcük AKBANKAYA özel değildir")


def test_case_insensitive():
    out = match_tickers("garanti bbva bugün hareketli")
    assert "GARAN.IS" in out


def test_empty_input():
    assert match_tickers("") == []
    assert match_tickers(None) == []
