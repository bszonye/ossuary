"""Unit tests for the __main__ module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from typing import TypeAlias

import pytest

import ossuary.__main__ as main

Cap: TypeAlias = pytest.CaptureFixture[str]


class TestMain:
    def test_main(self, capsys: Cap) -> None:
        argv: list[str] = []
        main.main(argv)
        cap = capsys.readouterr()
        assert cap.out == ""
        assert cap.err == ""


class TestEvalDemo:
    def test_empty(self, capsys: Cap) -> None:
        main.eval_demo([])
        cap = capsys.readouterr()
        assert cap.out == ""
        assert cap.err == ""

    def test_expressions(self, capsys: Cap) -> None:
        main.eval_demo(["None", "1", "1+1", "1+2"], interactive=False)
        cap = capsys.readouterr()
        assert cap.out == "1\n2\n3\n"
        assert cap.err == ""

    def test_pmf(self, capsys: Cap) -> None:
        main.eval_demo(["d2"], interactive=False)
        cap = capsys.readouterr()
        assert cap.out == "1  50.00\n2  50.00\n"
        assert cap.err == ""
