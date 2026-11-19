"""License/reference regressions, separate from dataset integrity implementation."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
MICROSOFT_REVISION = "cc042d12f6704be36c70f9f4b2457aeaa12b97c7"
INTEL_REVISION = "624952e3f420ae18d88b31977ee2ea436c833abb"
MIT_NOTICE = '''MIT License

Copyright (c) 2025 Azure AI Foundry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_complete_upstream_mit_notice_is_retained_with_sft_copies() -> None:
    folders = [
        ROOT / "Supervised_Fine_Tuning" / "Tool-Calling",
        ROOT / "Supervised_Fine_Tuning" / "Text-Bug-Detection",
        ROOT.parent / "managed_training" / "00-foundations"
        / "02-first-sft-bug-detection" / "data",
    ]
    for folder in folders:
        assert _read(folder / "LICENSE.md").strip() == MIT_NOTICE.strip()
        readme = _read(folder / "README.md")
        assert MICROSOFT_REVISION in readme
        assert "[LICENSE.md](LICENSE.md)" in readme


def test_sft_notices_pin_the_published_source_blobs_without_claiming_comparison() -> None:
    expected = {
        "Tool-Calling": (
            "b864adcdd3decff9e4332f572f51fd166c093ad1",
            "a22acedf0fc5e42d1a6bbf1a03eeb5c0bbe67691",
        ),
        "Text-Bug-Detection": (
            "fc9a739cbf84c5c3de3c5912fba087a51004c270",
            "d2a408dd90a0a7388ad6e338befc234c4d2f54d8",
        ),
    }
    for name, blobs in expected.items():
        text = _read(ROOT / "Supervised_Fine_Tuning" / name / "README.md")
        assert f"/tree/{MICROSOFT_REVISION}/Sample_Datasets/" in text
        assert f"/blob/{MICROSOFT_REVISION}/LICENSE" in text
        assert all(blob in text for blob in blobs)
        assert "not newly computed local" in text


def test_orca_notice_separates_intel_reference_from_historical_derivation() -> None:
    folder = ROOT / "Direct_Preference_Optimization" / "orca_dpo_pairs"
    text = _read(folder / "README.md")
    assert f"Intel/orca_dpo_pairs/tree/{INTEL_REVISION}" in text
    assert f"Intel/orca_dpo_pairs/raw/{INTEL_REVISION}/README.md" in text
    assert f"/tree/{MICROSOFT_REVISION}/Sample_Datasets/" in text
    assert "license: apache-2.0" in text
    assert "verified **reference snapshot**" in text
    assert "does not record the original Intel acquisition revision" in text
    assert "**not** a hash of either bundled" in text
    assert "no\nstandalone LICENSE or NOTICE file" in text
    assert "[Apache License 2.0](APACHE-2.0.txt)" in text
    assert "[Microsoft MIT notice](LICENSE.md)" in text
    scope = _read(folder / "LICENSE.md")
    assert "does not replace the Apache-2.0 declaration" in scope
    assert "// Copyright (c) Microsoft Corporation." in scope
    assert "The above copyright notice and this permission notice" in scope
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in scope


def test_retained_apache_license_includes_every_section_and_appendix() -> None:
    path = ROOT / "Direct_Preference_Optimization" / "orca_dpo_pairs" / "APACHE-2.0.txt"
    text = " ".join(_read(path).split())
    sections = (
        "1. Definitions.",
        "2. Grant of Copyright License.",
        "3. Grant of Patent License.",
        "4. Redistribution.",
        "5. Submission of Contributions.",
        "6. Trademarks.",
        "7. Disclaimer of Warranty.",
        "8. Limitation of Liability.",
        "9. Accepting Warranty or Additional Liability.",
        "END OF TERMS AND CONDITIONS",
        "APPENDIX: How to apply the Apache License to your work.",
    )
    assert "Version 2.0, January 2004" in text
    assert all(section in text for section in sections)
    assert "You must give any other recipients of the Work or Derivative Works a copy" in text
    assert 'If the Work includes a "NOTICE" text file' in text
    assert "Copyright [yyyy] [name of copyright owner]" in text
    assert text.endswith("limitations under the License.")