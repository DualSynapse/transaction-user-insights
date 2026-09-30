"""PDF components (reportlab): cover, headings, tables, images."""
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

_STYLES = getSampleStyleSheet()
_STYLES.add(ParagraphStyle(name="H1Custom", parent=_STYLES["Heading1"], spaceBefore=18, spaceAfter=10))
_STYLES.add(ParagraphStyle(name="H2Custom", parent=_STYLES["Heading2"], spaceBefore=12, spaceAfter=6))
_STYLES.add(ParagraphStyle(name="BodyCustom", parent=_STYLES["BodyText"], spaceAfter=8, leading=14))
_STYLES.add(ParagraphStyle(name="CoverTitle", parent=_STYLES["Title"], fontSize=26, spaceAfter=16))
_STYLES.add(ParagraphStyle(name="CoverSub", parent=_STYLES["Normal"], fontSize=12, alignment=1, spaceAfter=6))


def _clean(text: str) -> str:
    """Keep text inside Helvetica's Latin-1 range (avoid black-box glyphs)."""
    return (
        str(text)
        .replace("→", "->").replace("≥", ">=").replace("≤", "<=")
        .replace("–", "-").replace("—", "-").replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
    )


class ReportBuilder:
    def __init__(self, output_path, title: str, period: str):
        self.output_path = output_path
        self.story = []
        self.doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        )
        self._cover(title, period)

    def _cover(self, title: str, period: str):
        self.story.append(Spacer(1, 6 * cm))
        self.story.append(Paragraph(_clean(title), _STYLES["CoverTitle"]))
        self.story.append(Paragraph(_clean(f"Data period: {period}"), _STYLES["CoverSub"]))
        self.story.append(Paragraph(_clean(f"Generated on: {date.today().isoformat()}"), _STYLES["CoverSub"]))
        self.story.append(PageBreak())

    def h1(self, text: str):
        self.story.append(Paragraph(_clean(text), _STYLES["H1Custom"]))

    def h2(self, text: str):
        self.story.append(Paragraph(_clean(text), _STYLES["H2Custom"]))

    def p(self, text: str):
        self.story.append(Paragraph(_clean(text), _STYLES["BodyCustom"]))

    def bullets(self, items: list[str]):
        for item in items:
            self.p(f"• {item}" if False else f"- {item}")

    def image(self, path: str, width_cm: float = 15):
        self.story.append(Image(path, width=width_cm * cm, height=width_cm * cm * 0.55, kind="proportional"))
        self.story.append(Spacer(1, 0.3 * cm))

    def table(self, headers: list[str], rows: list[list], col_widths=None):
        data = [[_clean(h) for h in headers]] + [[_clean(c) for c in row] for row in rows]
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 0.4 * cm))

    def page_break(self):
        self.story.append(PageBreak())

    def spacer(self, height_cm: float = 0.5):
        self.story.append(Spacer(1, height_cm * cm))

    def build(self):
        self.doc.build(self.story)
