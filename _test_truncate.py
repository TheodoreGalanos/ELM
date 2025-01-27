"""
When there are unittests, please refactor this function into them.
"""
from codegen.codegen_utilities import truncate


def test_truncate():
    """
    Test `truncate` working with different non-positional arguments (assuming the truncation at terminal strings work).
    """
    t1 = "def test():\n\treturn 1\ndef test2():\n\treturn 2"
    t2 = "def test():\n\treturn 1"
    t3 = "def test():\n\treturn 1\ndef test2():\n\treturn 2\nif __name__ == '__main__':\n\t test()"
    t4 = "\n\treturn 2\nif __name__ == '__main__':\n\t test()"

    assert truncate(t1) == "def test():\n\treturn 1\n"
    assert truncate(t1, def_num=2) == t1
    assert truncate(t2) == t2
    assert truncate(t3) == "def test():\n\treturn 1\n"
    assert truncate(t3, def_num=2) == t3
    assert truncate(t4, only_local_scope=True) == "\n\treturn 2\n"


if __name__ == '__main__':
    test_truncate()
