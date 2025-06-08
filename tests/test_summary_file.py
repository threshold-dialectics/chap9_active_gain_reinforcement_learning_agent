import os, json


def test_trimmed_summary_exists():
    fname = os.path.join("results", "summary_proactive_stabilization.json")
    assert os.path.isfile(fname)
    with open(fname) as f:
        data = json.load(f)
    assert "meta" in data and "conditions" in data
