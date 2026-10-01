"""
Generate two small, fictional rulebook PDFs (two editions) for the quick start
and the tests. The content is invented; it only mimics the shape of a real
rulebook: numbered rules, running headers, page footers, rules that span pages,
and a rule that changes between editions.

    python examples/sample/make_sample_pdfs.py
"""

from pathlib import Path

HERE = Path(__file__).parent

RULES_2024 = [
    ("Rule 101", "Eligibility", [
        "A competitor must be a registered member of a national federation in good standing.",
        "Competitors must reach the minimum age of 13 by 1 July preceding the season to enter senior events.",
    ]),
    ("Rule 102", "Entries", [
        "Entries must be submitted by the national federation no later than 21 days before the event.",
        "Late entries may be accepted at the discretion of the organising committee.",
    ]),
    ("Rule 201", "Short Program - Required Elements", [
        "The short program consists of seven required elements.",
        "A maximum of three jump elements may be included: one axel type jump, one solo jump, "
        "and one jump combination.",
        "A jump combination may consist of two jumps. The second jump may be a toe loop or a loop.",
        "Any element beyond the permitted number will not be counted and receives no value.",
    ]),
    ("Rule 202", "Short Program - Duration", [
        "The duration of the short program must not exceed 2 minutes 40 seconds, plus or minus 10 seconds.",
        "Timing starts with the first movement of the competitor and ends when the competitor comes to a stop.",
    ]),
    ("Rule 301", "Free Skating - Well Balanced Program", [
        "A well balanced free skating program must contain a maximum of seven jump elements.",
        "One of the jump elements must be an axel type jump.",
        "A maximum of three jump combinations or sequences is permitted.",
        "A jump combination may contain up to three jumps; only one three-jump combination is permitted.",
        "A maximum of three spins is permitted, one of which must be a spin combination.",
        "Only two of the triple or quadruple jumps may be executed twice, either as solo jumps or as part "
        "of a combination or sequence. Repetition beyond this limit counts as a separate element with "
        "reduced value. " * 3,
    ]),
    ("Rule 302", "Free Skating - Duration", [
        "The duration of the free skating program must not exceed 4 minutes, plus or minus 10 seconds.",
    ]),
    ("Rule 401", "Costumes and Props", [
        "Clothing must be modest, dignified and appropriate for athletic competition.",
        "Props of any kind are not permitted. A deduction of 1.0 applies for a costume violation.",
    ]),
    ("Rule 501", "Falls and Interruptions", [
        "A fall is defined as loss of control by a competitor with the result that the majority of their "
        "body weight is on the ice supported by any other part of the body other than the blades.",
        "Each fall results in a deduction of 1.0 point. An interruption in excess of 10 seconds results in "
        "a deduction of 1.0 point.",
    ]),
]

# 2025-26 edition: Rule 202 duration changes and Rule 401 deduction changes
RULES_2025 = [r for r in RULES_2024 if r[0] not in ("Rule 202", "Rule 401")]
RULES_2025.insert(3, ("Rule 202", "Short Program - Duration", [
    "The duration of the short program must not exceed 2 minutes 50 seconds, plus or minus 10 seconds.",
    "Timing starts with the first movement of the competitor and ends when the competitor comes to a stop.",
]))
RULES_2025.insert(6, ("Rule 401", "Costumes and Props", [
    "Clothing must be modest, dignified and appropriate for athletic competition.",
    "Props of any kind are not permitted. A deduction of 2.0 applies for a costume violation.",
]))


def make_pdf(path: Path, edition: str, rules) -> None:
    import pymupdf

    doc = pymupdf.open()
    width, height = 595, 842
    margin, line_h, font_size = 60, 16, 10.5
    page, y = None, 0

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=width, height=height)
        page.insert_text((margin, 36), f"Sample Sport Technical Rules {edition}", fontsize=8)
        page.insert_text((width / 2 - 20, height - 30), f"Page {doc.page_count}", fontsize=8)
        y = 70

    def write(text: str, size: float = font_size, gap_after: float = 0):
        nonlocal y
        import textwrap
        for line in textwrap.wrap(text, 95) or [""]:
            if y > height - 70:
                new_page()
            page.insert_text((margin, y), line, fontsize=size)
            y += line_h
        y += gap_after

    new_page()
    write(f"SAMPLE SPORT TECHNICAL RULES - EDITION {edition}", size=14, gap_after=8)
    write("This fictional rulebook is used to demonstrate and test clippy_core. "
          "It is not a real rulebook.", gap_after=line_h)
    for i, (rule_id, title, paras) in enumerate(rules):
        if i in (3, 6):          # force some rules to start on a new page
            new_page()
        write(f"{rule_id} {title}", size=12, gap_after=4)
        for n, para in enumerate(paras, 1):
            write(f"{n}. {para}", gap_after=line_h * 0.6)
        y += line_h * 0.5
    doc.save(str(path))


def main():
    out = HERE / "pdfs"
    out.mkdir(exist_ok=True)
    make_pdf(out / "sample-rules-2024-25.pdf", "2024-25", RULES_2024)
    make_pdf(out / "sample-rules-2025-26.pdf", "2025-26", RULES_2025)
    print(f"Wrote sample PDFs to {out}")


if __name__ == "__main__":
    main()
