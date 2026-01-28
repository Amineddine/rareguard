from rare_guard.data import templateize

def test_templateize_basic():
    msg = "Failed password for invalid user admin from 10.1.2.3 port 22 ssh2"
    t = templateize(msg)
    assert "<IP>" in t
    assert "<NUM>" in t
