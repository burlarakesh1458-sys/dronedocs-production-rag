from dronedocs.loaders import clean_text, load_html, load_markdown


def test_clean_text_drops_devanagari_lines_and_table_rules():
    raw = "ड्रोन जनयम, 2021 का प्रारूप\nThe Drone Rules, 2021\n| --- | --- |\n| A | B |\nTHE GAZETTE OF INDIA : EXTRAORDINARY [PART II]"
    out = clean_text(raw)
    assert "The Drone Rules, 2021" in out
    assert "ड्रोन" not in out
    assert "---" not in out and "| A | B |" in out
    assert "GAZETTE" not in out


def test_markdown_split_by_heading_paths(tmp_path):
    p = tmp_path / "safety.md"
    p.write_text(
        "# Safety\nIntro text that is long enough to keep.\n\n## Battery {#bat}\n"
        "Battery failsafe text that is long enough.\n::: info\nnote\n:::\n<Badge text='x'/>"
    )
    secs = list(load_markdown(p))
    assert [s.section for s in secs] == ["Safety", "Safety > Battery"]
    assert "{#" not in secs[1].locator and "Badge" not in secs[1].text and ":::" not in secs[1].text


def test_html_tables_become_rows(tmp_path):
    p = tmp_path / "modes.html"
    p.write_text(
        "<html><body><div itemprop='articleBody'><h1>Modes</h1><p>Copter modes overview paragraph.</p>"
        "<table><tr><th>Mode</th><th>Summary</th></tr><tr><td>RTL</td><td>Returns home</td></tr></table>"
        "</div></body></html>"
    )
    secs = list(load_html(p))
    assert "RTL | Returns home" in secs[0].text
