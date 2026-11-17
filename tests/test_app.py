from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_views_and_filters():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value != "0"
    for chart in ["Sentiment distribution", "Top sources", "Headline word cloud"]:
        next(w for w in app.selectbox if w.label == "View").select(chart).run()
        assert not app.exception
    next(w for w in app.selectbox if w.label == "Concept method").select("Phrases (ngrams)").run()
    assert not app.exception
    source = app.sidebar.multiselect[0].options[0]
    app.sidebar.multiselect[0].select(source).run()
    assert not app.exception
    assert app.metric[1].value == "1"
    app.sidebar.multiselect[0].unselect(source).run()
    app.sidebar.multiselect[1].select("Positive").run()
    assert not app.exception
    assert any("Both positive and negative" in message.value for message in app.info)
    app.sidebar.text_input[0].input("no-match-987654321").run()
    assert not app.exception
    assert any("No articles match" in message.value for message in app.info)
