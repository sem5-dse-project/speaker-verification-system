"""Build the final presentation for the voice authentication project.

Design system
-------------
Direction   editorial / magazine
Palette     charcoal ink, forest green, warm amber, terracotta on warm paper
Type        Sitka Display (titles, hero figures) / Corbel (body) /
            Bahnschrift SemiBold Condensed (eyebrows, numerals)
Radius      one rounding scale, expressed in points, applied to every
            container, image mask and pill on every slide
Archetypes  hero, section divider, split (1/3 - 2/3), full, data

All diagrams are native PowerPoint shapes and all charts are native
PowerPoint charts, so the deck stays editable.

    python build_deck.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DERIVED = ASSETS / "derived"
OUT = HERE / "Voice-Auth-Final-Presentation.pptx"

# ------------------------------------------------------------------ palette
# One brand hue (orange) carries every piece of chrome. Emerald and crimson
# are reserved for data: they appear only inside tables, charts and hero
# figures, never as decoration, so a colour on a number always means
# something.
PAPER = RGBColor(0xFB, 0xF6, 0xEC)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1C, 0x1C, 0x1E)
MUTED = RGBColor(0x6B, 0x64, 0x5C)
FAINT = RGBColor(0xA9, 0xA1, 0x94)

# brand
ORANGE = RGBColor(0xEE, 0x72, 0x03)      # fills, bands, table heads
ORANGE_TX = RGBColor(0xB8, 0x4E, 0x08)   # small type: 4.7:1 on PAPER
WASH = RGBColor(0xFD, 0xEE, 0xDF)        # card fills, zebra rows
WASH_DEEP = RGBColor(0xFA, 0xDD, 0xC2)
HAIR = RGBColor(0xE8, 0xDF, 0xCE)

# data only
WIN = RGBColor(0x0C, 0x7C, 0x46)
FAIL = RGBColor(0xD4, 0x21, 0x3D)
NEUTRAL = RGBColor(0x8A, 0x83, 0x78)
SOLID_TX = RGBColor(0x4A, 0x2E, 0x0A)    # body type on an orange fill

DARK = RGBColor(0x17, 0x13, 0x0F)
D_INK = RGBColor(0xFB, 0xF6, 0xEC)
D_ORANGE = RGBColor(0xFF, 0x8C, 0x2A)
D_MUTED = RGBColor(0x9A, 0x90, 0x84)
D_PANEL = RGBColor(0x22, 0x1C, 0x16)
D_HAIR = RGBColor(0x3A, 0x31, 0x29)
D_WIN = RGBColor(0x2F, 0xBF, 0x6B)
D_FAIL = RGBColor(0xFF, 0x5E, 0x6E)

# ------------------------------------------------------------------ type
DISPLAY = "Sitka Display"
BANNER = "Sitka Banner"
BODY = "Corbel"
LABEL = "Bahnschrift SemiBold Condensed"

# ------------------------------------------------------------------ grid
SW, SH = 13.3333, 7.5
M = 0.75                      # page margin
CW = SW - 2 * M               # 11.833
RAIL_W = 3.95                 # narrow column
GUTTER = 0.60
MAIN_X = M + RAIL_W + GUTTER  # 5.30
MAIN_W = SW - M - MAIN_X      # 7.28
FOOT_Y = 6.95

R_CARD = 0.155                # ~11 pt corner radius
R_TILE = 0.12
R_PILL = 9.9                  # sentinel: full pill

SECTIONS = ["Problem", "System", "Business", "Research", "Close"]


# ------------------------------------------------------------------ helpers
def _in(v: float) -> Emu:
    return Inches(v)


def recolour(name: str, target: RGBColor, tol: int = 46) -> str:
    """Repaint a generated icon's flat backdrop so it blends with its host."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / f"{Path(name).stem}_{target[0]:02x}{target[1]:02x}{target[2]:02x}.png"
    if not out.exists():
        im = Image.open(ASSETS / name).convert("RGB")
        base = im.getpixel((2, 2))
        dst = (target[0], target[1], target[2])
        px = im.load()
        w, h = im.size
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                if abs(r - base[0]) + abs(g - base[1]) + abs(b - base[2]) <= tol:
                    px[x, y] = dst
        im.save(out)
    return str(out.relative_to(ASSETS)).replace("\\", "/")


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def set_bg(slide, colour: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def shape(slide, x, y, w, h, *, fill=None, line=None, lw=0.9,
          kind=MSO_SHAPE.RECTANGLE, radius=None):
    """Rounded corners are specified in inches and converted to the
    adjustment the shape needs, so the visual radius is identical
    regardless of the shape's size."""
    sp = slide.shapes.add_shape(kind, _in(x), _in(y), _in(w), _in(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sp.adjustments[0] = 0.5 if radius == R_PILL else min(
            0.5, radius / min(w, h))
    tf = sp.text_frame
    tf.word_wrap = True
    tf.margin_left = _in(0.16)
    tf.margin_right = _in(0.16)
    tf.margin_top = _in(0.06)
    tf.margin_bottom = _in(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sp


def panel(slide, x, y, w, h, *, fill=CARD, line=HAIR, radius=R_CARD):
    return shape(slide, x, y, w, h, fill=fill, line=line,
                 kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=radius)


def pill(slide, x, y, w, h, *, fill=ORANGE, line=None):
    return shape(slide, x, y, w, h, fill=fill, line=line,
                 kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=R_PILL)


def rule(slide, x, y, w, *, colour=HAIR, weight=0.012):
    return shape(slide, x, y, w, weight, fill=colour)


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(_in(x), _in(y), _in(w), _in(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


class W:
    """Small paragraph writer; reuses the first, empty paragraph."""

    def __init__(self, tf):
        self.tf = tf
        self.used = False

    def _p(self):
        if not self.used:
            self.used = True
            return self.tf.paragraphs[0]
        return self.tf.add_paragraph()

    @staticmethod
    def _style(run, size, colour, bold, font, track=None, italic=False):
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        # The brand orange only reaches 3:1 on cream, which fails for text at
        # reading size, so anything below display size uses the deep variant.
        run.font.color.rgb = ORANGE_TX if (colour == ORANGE and size < 20) \
            else colour
        run.font.name = font
        if track:
            run.font._rPr.set("spc", str(int(track * 100)))

    def eyebrow(self, text, *, colour=ORANGE, size=10, y_after=0):
        p = self._p()
        p.space_after = Pt(y_after)
        self._style(p.add_run(), size, colour, True, LABEL, track=1.6)
        p.runs[0].text = text.upper()
        return p

    def title(self, text, *, size=31, colour=INK, space_after=0,
              spacing=1.04, align=PP_ALIGN.LEFT, font=DISPLAY):
        p = self._p()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = spacing
        run = p.add_run()
        run.text = text
        self._style(run, size, colour, True, font)
        return p

    def body(self, text, *, size=13, colour=INK, bold=False, italic=False,
             space_after=7, spacing=1.18, align=PP_ALIGN.LEFT, font=BODY):
        p = self._p()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = spacing
        run = p.add_run()
        run.text = text
        self._style(run, size, colour, bold, font, italic=italic)
        return p

    def bullet(self, text, *, size=13, colour=INK, accent=ORANGE,
               space_after=10, spacing=1.18, indent=0.26, glyph="\u2014"):
        p = self._p()
        p.space_after = Pt(space_after)
        p.line_spacing = spacing
        pPr = p._p.get_or_add_pPr()
        emu = int(indent * 914400)
        pPr.set("marL", str(emu))
        pPr.set("indent", str(-emu))
        r1 = p.add_run()
        r1.text = f"{glyph}\t"
        self._style(r1, size, accent, True, BODY)
        r2 = p.add_run()
        r2.text = text
        self._style(r2, size, colour, False, BODY)
        return p

    def rich(self, parts, *, space_after=7, spacing=1.18,
             align=PP_ALIGN.LEFT):
        p = self._p()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = spacing
        for text, size, colour, bold, font in parts:
            run = p.add_run()
            run.text = text
            self._style(run, size, colour, bold, font)
        return p


def picture(slide, name, x, y, w, *, crop_t=0.0, crop_b=0.0,
            crop_l=0.0, crop_r=0.0, radius=None):
    path = ASSETS / name
    with Image.open(path) as im:
        sw, sh = im.size
    vw = sw * (1 - crop_l - crop_r)
    vh = sh * (1 - crop_t - crop_b)
    h = w * (vh / vw)
    pic = slide.shapes.add_picture(str(path), _in(x), _in(y), _in(w), _in(h))
    pic.crop_top, pic.crop_bottom = crop_t, crop_b
    pic.crop_left, pic.crop_right = crop_l, crop_r
    if radius:
        _round_picture(pic, radius, w, h)
    return pic, h


def _round_picture(pic, radius_in, w, h) -> None:
    spPr = pic._element.spPr
    old = spPr.find(qn("a:prstGeom"))
    if old is not None:
        spPr.remove(old)
    adj = int(min(0.5, radius_in / min(w, h)) * 100000)
    geom = parse_xml(
        '<a:prstGeom %s prst="roundRect"><a:avLst>'
        '<a:gd name="adj" fmla="val %d"/></a:avLst></a:prstGeom>'
        % (nsdecls("a"), adj))
    xfrm = spPr.find(qn("a:xfrm"))
    if xfrm is not None:
        xfrm.addnext(geom)
    else:
        spPr.insert(0, geom)


# ------------------------------------------------------------------ chrome
def band(slide, *, colour=ORANGE) -> None:
    """Full-bleed brand strip along the very top of the slide."""
    shape(slide, 0, 0, SW, 0.09, fill=colour)


def _progress(slide, section: int, number: int, dark: bool, *,
              hairline=None, muted=None, active=None) -> None:
    hairline = hairline or (D_HAIR if dark else HAIR)
    muted = muted or (D_MUTED if dark else FAINT)
    active = active or (D_ORANGE if dark else ORANGE)
    rule(slide, M, FOOT_Y, CW, colour=hairline)
    W(textbox(slide, M, FOOT_Y + 0.12, 6.0, 0.28)).body(
        "Voice-Based Speaker Verification for Secure User Authentication",
        size=8.5, colour=muted, font=BODY, spacing=1.0)
    for i in range(len(SECTIONS)):
        pill(slide, 10.30 + i * 0.34, FOOT_Y + 0.21, 0.28, 0.062,
             fill=active if i == section else hairline)
    W(textbox(slide, 12.05, FOOT_Y + 0.10, 0.53, 0.3)).body(
        f"{number:02d}", size=10.5, colour=muted, bold=True, font=LABEL,
        align=PP_ALIGN.RIGHT, spacing=1.0)


def slide_split(prs, eyebrow, title, lede, number, section):
    """Editorial 1/3 - 2/3 layout: title rail on the left, content right."""
    s = blank(prs)
    set_bg(s, PAPER)
    W(textbox(s, M, 0.72, RAIL_W, 0.26)).eyebrow(eyebrow)
    W(textbox(s, M, 1.02, RAIL_W, 2.05)).title(title)
    rule_y = 1.02 + (title.count("\n") + 1) * 0.455 + 0.42
    rule(s, M, rule_y, 1.45, colour=ORANGE, weight=0.045)
    if lede:
        W(textbox(s, M, rule_y + 0.26, RAIL_W, 2.4)).body(
            lede, size=12.5, colour=MUTED, spacing=1.34)
    band(s)
    _progress(s, section, number, False)
    return s


def slide_full(prs, eyebrow, title, number, section, *, lede=None):
    s = blank(prs)
    set_bg(s, PAPER)
    W(textbox(s, M, 0.62, 9.0, 0.26)).eyebrow(eyebrow)
    W(textbox(s, M, 0.90, 11.4, 0.62)).title(title)
    rule(s, M, 1.60, 1.45, colour=ORANGE, weight=0.045)
    if lede:
        W(textbox(s, M, 1.78, 10.4, 0.32)).body(
            lede, size=11.5, colour=MUTED, spacing=1.2)
    band(s)
    _progress(s, section, number, False)
    return s


def slide_divider(prs, index, name, line, number, section, *, tone="dark"):
    """Dividers alternate between near-black and full-bleed orange so the
    four section breaks read as a rhythm rather than a repeated template."""
    s = blank(prs)
    if tone == "orange":
        set_bg(s, ORANGE)
        ghost = RGBColor(0xD0, 0x63, 0x03)
        eb, ttl, accent, lead = SOLID_TX, INK, INK, INK
        bar_a, bar_b = INK, RGBColor(0xFF, 0xC4, 0x8A)
    else:
        set_bg(s, DARK)
        ghost = RGBColor(0x2E, 0x24, 0x1B)
        eb, ttl, accent, lead = D_ORANGE, D_INK, D_ORANGE, D_MUTED
        bar_a, bar_b = D_ORANGE, RGBColor(0x5A, 0x48, 0x38)
    W(textbox(s, M - 0.10, 1.05, 6.0, 3.1)).title(
        index, size=150, colour=ghost, font=BANNER)
    W(textbox(s, M, 3.62, 8.0, 0.3)).eyebrow(f"Section {index}", colour=eb)
    W(textbox(s, M, 3.96, 8.6, 0.9)).title(name, size=44, colour=ttl)
    rule(s, M, 5.02, 1.45, colour=accent, weight=0.05)
    W(textbox(s, M, 5.28, 7.6, 1.0)).body(
        line, size=14, colour=lead, spacing=1.3)

    heights = [.22, .34, .50, .70, .92, 1.10, 1.24, 1.30, 1.18, .98, .78,
               .58, .42, .30, .22]
    for i, hh in enumerate(heights):
        pill(s, 9.10 + i * 0.26, 3.60 - hh / 2, 0.115, hh,
             fill=bar_a if i == 7 else bar_b)
    if tone == "orange":
        _progress(s, section, number, False,
                  hairline=RGBColor(0xD0, 0x63, 0x03), muted=SOLID_TX,
                  active=INK)
    else:
        band(s, colour=D_ORANGE)
        _progress(s, section, number, True)
    return s


def slide_dark(prs, eyebrow, title, number, section):
    s = blank(prs)
    set_bg(s, DARK)
    W(textbox(s, M, 0.70, 9.0, 0.26)).eyebrow(eyebrow, colour=D_ORANGE)
    W(textbox(s, M, 1.00, 9.6, 0.62)).title(title, colour=D_INK)
    rule(s, M, 1.74, 1.45, colour=D_ORANGE, weight=0.045)
    band(s, colour=D_ORANGE)
    _progress(s, section, number, True)
    return s


# ------------------------------------------------------------------ widgets
def hero_number(slide, x, y, value, label, *, accent=ORANGE, sub=None,
                size=86, w=4.1):
    W(textbox(slide, x, y, w, size / 58)).title(
        value, size=size, colour=accent, spacing=0.92)
    yy = y + size / 68 + 0.12
    W(textbox(slide, x, yy, w, 0.34)).eyebrow(label, colour=INK, size=10.5)
    if sub:
        W(textbox(slide, x, yy + 0.30, w, 0.5)).body(
            sub, size=11, colour=MUTED, spacing=1.15)


def tile(slide, x, y, w, h, value, label, *, accent=ORANGE, dark=False,
         tone="wash"):
    if dark:
        fill, line, sub = D_PANEL, D_HAIR, D_MUTED
    elif tone == "wash":
        fill, line, sub = WASH, WASH_DEEP, SOLID_TX
    else:
        fill, line, sub = CARD, HAIR, MUTED
    panel(slide, x, y, w, h, fill=fill, line=line)
    wr = W(textbox(slide, x + 0.26, y + 0.20, w - 0.5, h - 0.36))
    wr.title(value, size=25, colour=accent, space_after=2)
    wr.body(label, size=10.5, colour=sub, spacing=1.12)


def note(slide, x, y, w, h, heading, text, *, accent=ORANGE, dark=False,
         tone="plain"):
    """tone: plain (white card) | wash (peach card) | solid (orange card).

    Solid cards set type in ink rather than white: dark on the brand orange
    reaches 6.5:1 where white would only manage 3:1.
    """
    if dark:
        fill, line, head_c, body_c = D_PANEL, D_HAIR, D_INK, D_MUTED
    elif tone == "solid":
        fill, line, head_c, body_c = ORANGE, None, INK, SOLID_TX
    elif tone == "wash":
        fill, line, head_c, body_c = WASH, WASH_DEEP, INK, SOLID_TX
    else:
        fill, line, head_c, body_c = CARD, HAIR, INK, MUTED
    panel(slide, x, y, w, h, fill=fill, line=line)
    pad = 0.30
    if tone != "solid":
        pill(slide, x + 0.22, y + 0.24, 0.055, h - 0.48, fill=accent)
        pad = 0.46
    wr = W(textbox(slide, x + pad, y + 0.16, w - pad - 0.26, h - 0.32,
                   anchor=MSO_ANCHOR.MIDDLE))
    wr.body(heading, size=12.5, colour=head_c, bold=True, space_after=5)
    wr.body(text, size=11, colour=body_c, spacing=1.24, space_after=0)


def table(slide, x, y, w, rows, *, col_w, size=12, row_h=0.46, head_h=0.44,
          head_fill=ORANGE_TX, head_ink=PAPER):
    n_rows, n_cols = len(rows), len(rows[0])
    total = head_h + row_h * (n_rows - 1)
    gfx = slide.shapes.add_table(n_rows, n_cols, _in(x), _in(y), _in(w),
                                 _in(total))
    tbl = gfx.table
    tbl.first_row = tbl.horz_banding = tbl.first_col = False
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = _in(cw)
    for r in range(n_rows):
        tbl.rows[r].height = _in(head_h if r == 0 else row_h)
    for r, row in enumerate(rows):
        head = r == 0
        for c, item in enumerate(row):
            cell = tbl.cell(r, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = _in(0.14)
            cell.margin_right = _in(0.10)
            cell.margin_top = cell.margin_bottom = _in(0.02)
            cell.fill.solid()
            cell.fill.fore_color.rgb = head_fill if head else (
                PAPER if r % 2 else WASH)
            if isinstance(item, dict):
                text = item.get("t", "")
                colour = item.get("c", head_ink if head else INK)
                bold = item.get("b", head)
                align = item.get("a", PP_ALIGN.LEFT)
                fsize = item.get("s", size)
                font = item.get("f", LABEL if head else BODY)
            else:
                text, colour, bold = item, (head_ink if head else INK), head
                align, fsize = PP_ALIGN.LEFT, size
                font = LABEL if head else BODY
            p = cell.text_frame.paragraphs[0]
            p.alignment = align
            run = p.add_run()
            run.text = text
            run.font.size = Pt(fsize + (0.5 if head else 0))
            run.font.bold = bold
            run.font.color.rgb = colour
            run.font.name = font
    return total


# ------------------------------------------------------------------ charts
def _strip_chart_chrome(chart) -> None:
    chart.has_title = False
    chart.font.name = BODY
    chart.font.size = Pt(10.5)
    chart.font.color.rgb = MUTED
    cs = chart._chartSpace
    rc = cs.find(qn("c:roundedCorners"))
    if rc is not None:
        rc.set("val", "0")
    nofill = '<a:noFill/><a:ln><a:noFill/></a:ln>'
    chart_el = cs.find(qn("c:chart"))
    chart_el.addnext(parse_xml(
        "<c:spPr %s>%s</c:spPr>" % (nsdecls("c", "a"), nofill)))
    plot_area = chart_el.find(qn("c:plotArea"))
    plot_area.append(parse_xml(
        "<c:spPr %s>%s</c:spPr>" % (nsdecls("c", "a"), nofill)))


def _axes(chart, *, cat_size=10.5) -> None:
    va = chart.value_axis
    va.has_major_gridlines = False
    va.visible = False
    va.major_tick_mark = XL_TICK_MARK.NONE
    ca = chart.category_axis
    ca.has_major_gridlines = False
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.format.line.color.rgb = HAIR
    ca.tick_labels.font.size = Pt(cat_size)
    ca.tick_labels.font.name = BODY
    ca.tick_labels.font.color.rgb = INK


def _labels(plot, *, size=11, bold=True, colour=INK, fmt='0.00"%"'):
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(size)
    dl.font.bold = bold
    dl.font.name = BODY
    dl.font.color.rgb = colour


def column_chart(slide, x, y, w, h, categories, series, *, gap=60,
                 label_size=10.5, fmt='0.00"%"', point_colours=None):
    data = CategoryChartData()
    data.categories = categories
    for name, values, _ in series:
        data.add_series(name, values)
    gfx = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, _in(x),
                                 _in(y), _in(w), _in(h), data)
    chart = gfx.chart
    _strip_chart_chrome(chart)
    _axes(chart)
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.overlap = -10
    _labels(plot, size=label_size, fmt=fmt)
    for i, (_, _, colour) in enumerate(series):
        ser = chart.series[i]
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = colour
        ser.format.line.fill.background()
    if point_colours:
        plot.vary_by_categories = True
        ser = chart.series[0]
        for i, colour in enumerate(point_colours):
            pt = ser.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = colour
            pt.format.line.fill.background()
    if len(series) > 1:
        chart.has_legend = True
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(10.5)
        chart.legend.font.name = BODY
        chart.legend.font.color.rgb = MUTED
    else:
        chart.has_legend = False
    return chart


def bar_chart(slide, x, y, w, h, categories, values, colours, *, gap=55):
    data = CategoryChartData()
    data.categories = categories
    data.add_series("EER", values)
    gfx = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, _in(x), _in(y),
                                 _in(w), _in(h), data)
    chart = gfx.chart
    _strip_chart_chrome(chart)
    _axes(chart, cat_size=11)
    chart.has_legend = False
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.vary_by_categories = True
    _labels(plot, size=11.5)
    ser = chart.series[0]
    for i, colour in enumerate(colours):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = colour
        pt.format.line.fill.background()
    return chart


# ------------------------------------------------------------------ deck
def build() -> None:
    prs = Presentation()
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)
    nav = {}

    ic = {n: recolour(f"v3_icon_{n}.png", PAPER)
          for n in ("mic", "shield", "enhance", "database", "deploy", "fusion")}
    icw = {n: recolour(f"v3_icon_{n}.png", CARD)
           for n in ("mic", "shield", "enhance", "deploy")}

    # ---------------------------------------------------------- 01  hero
    s = blank(prs)
    set_bg(s, DARK)
    picture(s, recolour("v3_hero_title.png", DARK), 5.05, 1.42, 8.00)
    W(textbox(s, 0.95, 1.52, 3.9, 0.3)).eyebrow("Final project presentation",
                                                colour=D_ORANGE, size=11)
    W(textbox(s, 0.95, 1.92, 4.05, 2.3)).title(
        "Voice-Based Speaker Verification", size=40, colour=D_INK,
        spacing=1.0)
    rule(s, 0.95, 4.30, 1.45, colour=D_ORANGE, weight=0.05)
    W(textbox(s, 0.95, 4.56, 4.05, 0.95)).body(
        "Secure user authentication under spoofing and noise",
        size=14.5, colour=D_ORANGE, spacing=1.3)
    wr = W(textbox(s, 0.95, 5.66, 4.4, 1.0))
    for nm in ("Muditha Herath", "Yasith Hewarathna", "Herath H.M.M.P.B"):
        wr.body(nm, size=12.5, colour=D_INK, space_after=3, spacing=1.05)

    # ---------------------------------------------------------- 02  divider
    slide_divider(prs, "01", "The problem",
                  "Why a voice alone is not enough to prove who you are.",
                  2, 0)

    # ---------------------------------------------------------- 03  problem
    s = slide_split(
        prs, "The problem", "Why voice\nauthentication\nis hard",
        "Voice is the most convenient credential we have. It is also the "
        "easiest one to record, and the first to break when the room gets "
        "loud.", 3, 0)
    picture(s, "v3_problem.png", MAIN_X, 0.92, MAIN_W, radius=R_CARD)

    risks = [("Risk 01", "Spoof and replay",
              "An attacker plays back a recording of the real user.", ORANGE),
             ("Risk 02", "Background noise",
              "Street, office and babble noise break the match.", ORANGE)]
    cx = MAIN_X
    for tag, title, body, accent in risks:
        panel(s, cx, 5.08, 3.54, 1.52)
        pill(s, cx + 0.24, 5.32, 0.055, 1.04, fill=accent)
        wr = W(textbox(s, cx + 0.48, 5.30, 2.90, 1.15))
        wr.eyebrow(tag, colour=accent, size=9, y_after=4)
        wr.body(title, size=13, colour=INK, bold=True, space_after=4)
        wr.body(body, size=10.5, colour=MUTED, spacing=1.2)
        cx += 3.74

    # ---------------------------------------------------------- 04  metrics
    s = slide_split(
        prs, "Definitions", "What we\nmeasure",
        "Equal Error Rate, or EER, is the operating point where false "
        "accepts and false rejects are equal. Lower is better.", 4, 0)
    picture(s, ic["shield"], M + 0.05, 5.42, 1.30)

    defs = [("SV", "Is this the claimed speaker?",
             "Compares a probe against one enrolled template.", ORANGE),
            ("Anti-spoof", "Is this live speech or a fake?",
             "Flags replayed and synthetic audio before scoring.", ORANGE),
            ("SASV", "Both checks in a single score",
             "Accepts only a genuine speaker speaking live.", ORANGE)]
    yy = 1.02
    for term, question, detail, accent in defs:
        panel(s, MAIN_X, yy, MAIN_W, 1.36)
        pill(s, MAIN_X + 0.26, yy + 0.22, 0.055, 0.92, fill=accent)
        wr = W(textbox(s, MAIN_X + 0.52, yy + 0.22, MAIN_W - 0.85, 1.0))
        wr.eyebrow(term, colour=accent, size=9.5, y_after=5)
        wr.body(question, size=14, colour=INK, bold=True, space_after=4)
        wr.body(detail, size=11, colour=MUTED, spacing=1.2)
        yy += 1.52
    note(s, MAIN_X, 5.62, MAIN_W, 0.98,
         "Protocol discipline",
         "Thresholds are tuned on the development set. The evaluation set is "
         "reported once, with no further tuning.", tone="solid")

    # ---------------------------------------------------------- 05  value
    s = slide_full(prs, "Motivation", "Why this matters", 5, 0,
                   lede="Four reasons this is worth building, and worth "
                        "trusting.")
    cards = [(icw["mic"], "Frictionless", "No password typing and no code to "
              "wait for on repeat logins.", ORANGE),
             (icw["shield"], "Spoof aware", "A recorded voice is detected and "
              "rejected before any scoring.", ORANGE),
             (icw["enhance"], "Noise ready", "Enhancement and learned fusion "
              "hold accuracy up in noise.", ORANGE),
             (icw["deploy"], "Deployed", "A running web stack with automated "
              "build and deploy to cloud.", ORANGE)]
    cx = M
    for img, title, body, accent in cards:
        panel(s, cx, 2.28, 2.80, 3.30)
        pill(s, cx, 2.28, 2.80, 0.075, fill=accent)
        picture(s, img, cx + 0.90, 2.66, 1.00)
        wr = W(textbox(s, cx + 0.30, 3.92, 2.20, 1.42))
        wr.body(title, size=15, colour=INK, bold=True, space_after=7,
                align=PP_ALIGN.CENTER, font=DISPLAY)
        wr.body(body, size=11, colour=MUTED, align=PP_ALIGN.CENTER,
                spacing=1.28)
        cx += 3.01
    W(textbox(s, M, 6.06, CW, 0.4)).body(
        "Fintech  \u00b7  call centres  \u00b7  access control  \u00b7  "
        "passwordless consumer apps", size=12, colour=ORANGE, bold=True,
        align=PP_ALIGN.CENTER, font=LABEL)

    # ---------------------------------------------------------- 06  divider
    slide_divider(prs, "02", "The system",
                  "One architecture from the browser to the speaker model.",
                  6, 1, tone="orange")

    # ---------------------------------------------------------- 07  arch
    s = slide_dark(prs, "System design", "End-to-end architecture", 7, 1)

    def dbox(x, y, w, h, title, sub, owner, accent):
        panel(s, x, y, w, h, fill=D_PANEL, line=D_HAIR)
        pill(s, x, y, w, 0.065, fill=accent)
        wr = W(textbox(s, x + 0.24, y + 0.22, w - 0.48, h - 0.36))
        wr.body(title, size=12.5, colour=D_INK, bold=True, space_after=4,
                font=DISPLAY)
        wr.body(sub, size=9.5, colour=D_MUTED, space_after=4, spacing=1.16)
        if owner:
            wr.eyebrow(owner, colour=accent, size=8.5)

    def arrow(x, y, w, h, kind):
        shape(s, x, y, w, h, fill=D_ORANGE, kind=kind)

    dbox(M, 2.10, 3.10, 1.26, "Client  \u00b7  React",
         "Record  \u00b7  enroll  \u00b7  verify  \u00b7  1:N login",
         "Muditha", D_ORANGE)
    arrow(3.98, 2.54, 0.52, 0.36, MSO_SHAPE.RIGHT_ARROW)
    dbox(4.64, 2.10, 3.10, 1.26, "Backend  \u00b7  Express",
         "JWT auth  \u00b7  uploads  \u00b7  orchestration", "Muditha", D_ORANGE)
    arrow(7.90, 2.54, 0.52, 0.36, MSO_SHAPE.RIGHT_ARROW)

    arrow(2.08, 3.44, 0.34, 0.42, MSO_SHAPE.DOWN_ARROW)
    dbox(M, 3.96, 3.10, 1.26, "Enrollment",
         "N clips \u2192 VAD \u2192 embed \u2192 L2-norm \u2192 average", "",
         D_ORANGE)
    arrow(6.02, 3.44, 0.34, 0.42, MSO_SHAPE.DOWN_ARROW)
    dbox(4.64, 3.96, 3.10, 1.26, "PostgreSQL + pgvector",
         "vector(192) templates  \u00b7  HNSW cosine", "Muditha", D_ORANGE)

    panel(s, 8.56, 2.10, 4.02, 3.98, fill=D_PANEL,
          line=RGBColor(0x6B, 0x45, 0x1F))
    W(textbox(s, 8.80, 2.28, 3.6, 0.3)).eyebrow("ML server  \u00b7  FastAPI",
                                                colour=D_ORANGE, size=10)
    steps = [("01  Silero VAD", "keep speech, drop silence", D_ORANGE),
             ("02  Anti-spoof", "high risk \u2192 reject early", D_ORANGE),
             ("03  Enhancement", "WebRTC / Wave-U-Net", D_ORANGE),
             ("04  ECAPA \u00b1 TitaNet", "192-D speaker embedding", D_ORANGE),
             ("05  Fusion", "blend raw and enhanced", D_ORANGE),
             ("06  Cosine + threshold", "accept / reject + scores", D_ORANGE)]
    sy = 2.68
    for label, sub, accent in steps:
        panel(s, 8.80, sy, 3.54, 0.52, fill=RGBColor(0x2B, 0x23, 0x1B),
              line=None, radius=R_TILE)
        pill(s, 8.96, sy + 0.12, 0.05, 0.28, fill=accent)
        wr = W(textbox(s, 9.16, sy + 0.07, 3.10, 0.40))
        wr.body(label, size=10, colour=D_INK, bold=True, space_after=0,
                spacing=1.0)
        wr.body(sub, size=8.5, colour=D_MUTED, spacing=1.0)
        sy += 0.57

    panel(s, M, 6.16, CW, 0.52, fill=RGBColor(0x2E, 0x1C, 0x0B),
          line=RGBColor(0x7A, 0x4A, 0x18), radius=R_TILE)
    W(textbox(s, 1.02, 6.31, 11.3, 0.3)).rich([
        ("DEPLOY     ", 10, D_ORANGE, True, LABEL),
        ("GitHub main  \u2192  build images (GHCR)  \u2192  SSH deploy  "
         "\u2192  docker compose pull && up  on EC2", 10.5, D_ORANGE, False,
         BODY)], spacing=1.0)

    # ---------------------------------------------------------- 08  pipeline
    s = slide_full(prs, "How it works", "The verification pipeline", 8, 1,
                   lede="Nine steps from a microphone to an accept or reject "
                        "decision.")
    steps = [("01", "Capture", "Browser recording or uploaded WAV", ORANGE),
             ("02", "Normalise", "16 kHz, mono, float32", ORANGE),
             ("03", "Silero VAD", "Speech only; reject silent clips", ORANGE),
             ("04", "Anti-spoof", "Replay risk \u2192 reject early", ORANGE),
             ("05", "Enhancement", "WebRTC or Wave-U-Net", ORANGE),
             ("06", "Embedding", "ECAPA-TDNN, 192 dimensions", ORANGE),
             ("07", "Fusion", "Learned blend, raw and enhanced", ORANGE),
             ("08", "Cosine", "Score against the template", ORANGE),
             ("09", "Threshold", "Accept or reject, score returned", ORANGE)]
    cw_, ch_, gx, gy = 3.78, 1.24, 0.25, 0.20
    for i, (num, title, sub, accent) in enumerate(steps):
        x = M + (i % 3) * (cw_ + gx)
        y = 2.12 + (i // 3) * (ch_ + gy)
        panel(s, x, y, cw_, ch_)
        shape(s, x + 0.26, y + 0.30, 0.64, 0.64, fill=accent,
              kind=MSO_SHAPE.OVAL)
        W(textbox(s, x + 0.26, y + 0.47, 0.64, 0.3)).body(
            num, size=12.5, colour=PAPER, bold=True, align=PP_ALIGN.CENTER,
            font=LABEL, spacing=1.0)
        wr = W(textbox(s, x + 1.04, y + 0.28, cw_ - 1.30, 0.85))
        wr.body(title, size=13.5, colour=INK, bold=True, space_after=3,
                font=DISPLAY)
        wr.body(sub, size=10.5, colour=MUTED, spacing=1.14)
    W(textbox(s, M, 6.44, CW, 0.4)).rich([
        ("ENROLLMENT     ", 10, ORANGE, True, LABEL),
        ("several utterances  \u2192  VAD  \u2192  embed  \u2192  "
         "L2-normalise  \u2192  average  \u2192  store in pgvector",
         11.5, MUTED, False, BODY)])

    # ---------------------------------------------------------- 09  stack
    s = slide_full(prs, "Implementation", "Technology stack", 9, 1,
                   lede="A conventional web stack in front of a speech model "
                        "server.")
    table(s, M, 2.16, 5.75,
          [["Application layer", "Technology"],
           ["Frontend", "React + Vite, MediaRecorder"],
           ["Backend", "Node.js / Express, JWT, Multer"],
           ["Database", "PostgreSQL + pgvector, HNSW"],
           ["API surface", "/enroll/template, /verify, /replay"],
           ["Deployment", "Docker Compose, GHCR, Actions"]],
          col_w=[1.95, 3.80], row_h=0.56, head_h=0.46, size=11.5)
    table(s, 6.83, 2.16, 5.75,
          [["Model layer", "Technology"],
           ["Speech gating", "Silero VAD (ONNX)"],
           ["Speaker encoder", "SpeechBrain ECAPA-TDNN"],
           ["Second encoder", "NVIDIA NeMo TitaNet-Large"],
           ["Enhancement", "WebRTC NS, tuned Wave-U-Net"],
           ["Anti-spoof", "Replay CNN, AASIST, LFCC"]],
          col_w=[1.95, 3.80], row_h=0.56, head_h=0.46, size=11.5,
          head_fill=INK)
    note(s, M, 5.72, CW, 0.94, "One contract across the whole system",
         "Audio is always 16 kHz mono float32 and embeddings are always 192-D "
         "and L2-normalised, so the product path and the research notebooks "
         "score the same way.", tone="solid")

    # ---------------------------------------------------------- 10  choices
    s = slide_full(prs, "Implementation", "Key design decisions", 10, 1,
                   lede="Six choices that shaped the system, and the reason "
                        "behind each.")
    rows = [("Average several enrollment clips",
             "A single utterance overfits to one recording session.", ORANGE),
            ("Run VAD before scoring",
             "Silence and room tone corrupt the embedding.", ORANGE),
            ("Never enhance blindly",
             "Enhancement can hurt, so the fusion learns when to trust it.",
             ORANGE),
            ("Put the spoof gate inside the decision",
             "Security first: reject replay before identity matching.", ORANGE),
            ("Two encoder slots in the database",
             "ECAPA and TitaNet templates live side by side.", ORANGE),
            ("Compose deploy instead of k3s",
             "Fits a small, low-memory cloud instance.", ORANGE)]
    for i, (title, why, accent) in enumerate(rows):
        x = M + (i % 2) * 6.08
        y = 2.22 + (i // 2) * 1.50
        panel(s, x, y, 5.75, 1.34)
        pill(s, x + 0.24, y + 0.30, 0.055, 0.74, fill=accent)
        wr = W(textbox(s, x + 0.50, y + 0.16, 5.05, 1.02,
                       anchor=MSO_ANCHOR.MIDDLE))
        wr.body(title, size=13, colour=INK, bold=True, space_after=4,
                font=DISPLAY)
        wr.body(why, size=10.5, colour=MUTED, spacing=1.18, space_after=0)

    # ---------------------------------------------------------- 11  divider
    slide_divider(prs, "03", "The business",
                  "Who pays for this, and what they measure it by.", 11, 2)

    # ---------------------------------------------------------- 12  value
    s = slide_split(
        prs, "Business track", "Where this\ncreates value",
        "The same pipeline serves a bank, a call centre and a door. What "
        "changes between them is only the risk threshold.", 12, 2)
    picture(s, "v3_business.png", MAIN_X, 0.98, MAIN_W, crop_t=0.11,
            crop_b=0.11, radius=R_CARD)
    cols = [("Who buys it", ["Banks and fintech apps", "Call-centre identity",
                             "Building and device access",
                             "Passwordless consumer apps"], ORANGE),
            ("What they get", ["Fewer steps per login",
                               "Replay-attack resistance",
                               "Works in noisy channels",
                               "On-premise or cloud"], ORANGE),
            ("How we ship it", ["Enroll, verify, identify API",
                                "Threshold and risk dashboard",
                                "Pilot on one workflow",
                                "Tuned per customer"], ORANGE)]
    cx = MAIN_X
    for title, items, accent in cols:
        panel(s, cx, 4.32, 2.29, 2.28)
        pill(s, cx, 4.32, 2.29, 0.07, fill=accent)
        wr = W(textbox(s, cx + 0.26, 4.56, 1.80, 1.90))
        wr.eyebrow(title, colour=accent, size=9.5, y_after=9)
        for it in items:
            wr.bullet(it, size=10.5, colour=MUTED, accent=accent,
                      space_after=8, indent=0.17, glyph="\u00b7", spacing=1.1)
        cx += 2.50

    # ---------------------------------------------------------- 13  kpis
    s = slide_split(
        prs, "Business track", "The numbers\na customer\nasks for",
        "Every claim below comes from a run in this repository, including "
        "the ones that did not go our way.", 13, 2)
    table(s, MAIN_X, 0.98, MAIN_W,
          [["KPI", "Why the business cares"],
           ["EER / FAR / FRR", "Security versus user friction"],
           ["Spoof catch rate", "Direct fraud prevention"],
           ["Verification latency", "Perceived speed of the login"],
           ["Cost per verification", "Unit economics at scale"],
           ["Enrollment time", "Onboarding drop-off"]],
          col_w=[2.60, 4.68], row_h=0.46, head_h=0.42, size=11.5)
    W(textbox(s, MAIN_X, 3.92, MAIN_W, 0.3)).eyebrow("Evidence we already have")
    ev = [("0.83%", "SASV-EER, clean", WIN),
          ("8.79%", "all-noisy SV-EER", WIN),
          ("0.54%", "VoxCeleb EER", WIN),
          ("Live", "cloud deployment", INK)]
    cx = MAIN_X
    for value, label, accent in ev:
        tile(s, cx, 4.28, 1.70, 1.10, value, label, accent=accent)
        cx += 1.86
    note(s, MAIN_X, 5.54, MAIN_W, 1.06, "Responsible use",
         "Biometric consent and a retention policy are mandatory, and "
         "high-risk transactions should still use a second factor.",
         accent=ORANGE)

    # ---------------------------------------------------------- 14  divider
    slide_divider(prs, "04", "The research",
                  "Three tracks, three protocols, and one result we did not "
                  "want.", 14, 3, tone="orange")

    # ---------------------------------------------------------- 15  team
    s = slide_split(
        prs, "Research track", "Who built\nwhat",
        "Identity, robustness and security were developed and measured "
        "separately, then integrated into one pipeline.", 15, 3)
    picture(s, "v3_team.png", MAIN_X, 0.98, MAIN_W, crop_t=0.05, crop_b=0.05,
            radius=R_CARD)
    people = [("Muditha Herath", "Speaker models, Silero VAD, pgvector and "
               "the full app and ML integration", ORANGE),
              ("Yasith Hewarathna", "Noise resilience, enhancer comparison, "
               "Wave-U-Net and learned fusion", ORANGE),
              ("Herath H.M.M.P.B", "Replay and spoof detection, SASV "
               "protocol, noise-gated SASV and CI/CD", ORANGE)]
    yy = 4.28
    for i, (name, detail, accent) in enumerate(people):
        sp = panel(s, MAIN_X, yy, MAIN_W, 0.76)
        pill(s, MAIN_X + 0.24, yy + 0.18, 0.055, 0.40, fill=accent)
        wr = W(textbox(s, MAIN_X + 0.50, yy + 0.14, MAIN_W - 0.80, 0.55))
        wr.body(name, size=12.5, colour=INK, bold=True, space_after=2,
                font=DISPLAY)
        wr.body(detail, size=10, colour=MUTED, spacing=1.08)
        nav[i] = sp
        yy += 0.86
    W(textbox(s, M, 5.86, RAIL_W, 0.6)).body(
        "The cards on the right are clickable.", size=10.5, colour=FAINT,
        italic=True)

    # ---------------------------------------------------------- 16  muditha
    s = slide_split(
        prs, "Muditha Herath", "Speaker core\nand product",
        "The part of the project that actually runs in production: capture "
        "the voice, gate it, turn it into a vector, and search it.", 16, 3)
    picture(s, ic["mic"], M + 0.05, 5.26, 1.25)
    picture(s, ic["database"], M + 1.45, 5.26, 1.25)

    wr = W(textbox(s, MAIN_X, 1.00, MAIN_W, 3.4))
    for t in ("Integrated ECAPA-TDNN and NVIDIA TitaNet-Large into the ML "
              "server, with offline checkpoint loading",
              "Added Silero VAD to the live inference path, including "
              "zero-speech fallback handling",
              "Migrated storage to PostgreSQL + pgvector with HNSW cosine "
              "indexes for 1:1 and 1:N search",
              "Built the Express API, the React recorder and the "
              "verification result view",
              "Authored the VAD, audio and schema test suites with pytest "
              "markers"):
        wr.bullet(t, size=13, space_after=20)
    note(s, MAIN_X, 4.60, MAIN_W, 2.00, "Why a dual-encoder design",
         "ECAPA is small, open and fast on CPU, which suits the deployed "
         "service. TitaNet is kept as a second template slot so a stronger "
         "encoder can be swapped in later without a schema change.",
         tone="wash")

    # ---------------------------------------------------------- 17  vad
    s = slide_full(prs, "Muditha Herath", "Does removing silence help?", 17, 3,
                   lede="VoxCeleb1 test speakers, 1,120 same-speaker and "
                        "1,120 different-speaker pairs over 320 files.")
    hero_number(s, M, 2.30, "0.54%", "EER with Silero VAD", accent=WIN,
                sub="down from 0.625% on raw audio, a gain of 0.09 points")
    tile(s, M, 4.72, 2.02, 1.10, "92%", "speech kept", accent=ORANGE)
    tile(s, 2.90, 4.72, 2.02, 1.10, "320", "trial files", accent=INK)

    column_chart(s, 5.20, 2.22, 3.40, 3.60,
                 ["No VAD", "Silero VAD"],
                 [("EER %", (0.625, 0.536), NEUTRAL)], gap=95,
                 fmt='0.000"%"', point_colours=[NEUTRAL, WIN])
    note(s, 8.95, 2.22, 3.63, 3.60, "Reading the result honestly",
         "The EER gain is small, because VoxCeleb clips are already almost "
         "all speech.\n\nThe bigger practical win is robustness: silent or "
         "empty recordings are now caught instead of producing a meaningless "
         "embedding.\n\nThe TitaNet comparison runs on the same trial list "
         "and is blocked only on the NeMo runtime install.", tone="wash")

    # ---------------------------------------------------------- 18  yasith
    s = slide_split(
        prs, "Yasith Hewarathna", "Noise-robust\nverification",
        "Babble is the hardest condition in the whole study, and generic "
        "enhancement often makes verification worse rather than better.",
        18, 3)
    picture(s, "v3_research.png", MAIN_X, 0.98, MAIN_W, crop_t=0.10,
            crop_b=0.12, radius=R_CARD)
    ideas = [("The idea", "Do not choose between raw and enhanced audio. "
              "Embed both with ECAPA and let a network learn how to blend "
              "them per utterance.", ORANGE),
             ("The final system", "A Wave-U-Net enhancer fine-tuned with five "
              "speaker-aware losses, followed by self-attention fusion.",
              ORANGE)]
    cx = MAIN_X
    for title, body, accent in ideas:
        panel(s, cx, 4.30, 3.54, 2.30)
        pill(s, cx, 4.30, 3.54, 0.07, fill=accent)
        wr = W(textbox(s, cx + 0.30, 4.44, 2.94, 2.02,
                       anchor=MSO_ANCHOR.MIDDLE))
        wr.body(title, size=14, colour=INK, bold=True, space_after=7,
                font=DISPLAY)
        wr.body(body, size=11, colour=MUTED, spacing=1.26, space_after=0)
        cx += 3.74

    # ---------------------------------------------------------- 19  phases
    s = slide_full(prs, "Yasith Hewarathna", "From a fast baseline to a tuned "
                   "enhancer", 19, 3)
    phases = [("01", "Baselines", "ECAPA beats x-vector everywhere; Sinhala "
               "is far harder than English on the same encoder.", ORANGE),
              ("02", "Noise study", "Fourteen noise and reverb conditions. "
               "Babble at low SNR dominates the failures.", ORANGE),
              ("03", "Fusion MVP", "WebRTC enhancement and 524k paired "
               "embeddings. Attention fusion beats both baselines.", ORANGE),
              ("04", "Bake-off", "Eight enhancers compared end to end; only "
               "a task-tuned one helps on average.", ORANGE),
              ("05", "Final system", "Wave-U-Net fine-tuned with five losses, "
               "plus self-attention fusion.", ORANGE)]
    w_, pitch = 2.15, 2.42
    for i, (tag, title, body, accent) in enumerate(phases):
        x = M + i * pitch
        panel(s, x, 2.16, w_, 2.44)
        pill(s, x, 2.16, w_, 0.07, fill=accent)
        wr = W(textbox(s, x + 0.26, 2.32, w_ - 0.50, 2.16,
                       anchor=MSO_ANCHOR.MIDDLE))
        wr.eyebrow(f"Phase {tag}", colour=accent, size=9, y_after=6)
        wr.body(title, size=13.5, colour=INK, bold=True, space_after=6,
                font=DISPLAY)
        wr.body(body, size=10, colour=MUTED, spacing=1.2, space_after=0)
        if i < len(phases) - 1:
            shape(s, x + w_ + 0.05, 3.27, 0.18, 0.22, fill=HAIR,
                  kind=MSO_SHAPE.RIGHT_ARROW)
    note(s, M, 4.90, CW, 1.50, "The negative result that shaped the design",
         "DeepFilterNet3 enhancement helped only on babble at moderate SNR "
         "and hurt on clean, reverberant and stationary-noise audio. That is "
         "exactly why the final system learns a blend instead of always "
         "enhancing.", tone="solid")

    # ---------------------------------------------------------- 20  results
    s = slide_full(prs, "Yasith Hewarathna", "Final results on VoxCeleb1 + "
                   "MUSAN", 20, 3,
                   lede="37,611 trials, three noise types across nine SNR "
                        "levels plus clean. EER in percent, lower is better.")
    hero_number(s, M, 2.32, "8.79%", "all-noisy EER after fusion",
                accent=WIN,
                sub="down from 12.58% on raw ECAPA, a gain of 3.78 points")
    tile(s, M, 4.78, 1.90, 1.10, "+0.03", "clean EER cost", accent=MUTED)
    tile(s, 2.78, 4.78, 1.90, 1.10, "14.62%", "hardest subset", accent=ORANGE)

    column_chart(s, 5.10, 2.18, 7.48, 3.70,
                 ["Clean", "Babble", "Music", "Noise", "All noisy"],
                 [("Raw ECAPA", (0.90, 18.41, 12.64, 6.68, 12.58), NEUTRAL),
                   ("Enhancer + fusion", (0.94, 10.99, 10.20, 5.19, 8.79),
                   WIN)], gap=42, label_size=8.5)
    W(textbox(s, 5.10, 6.05, 7.48, 0.5)).body(
        "The fused system wins at every SNR and every noise type, with no "
        "crossover, and gives up almost nothing on clean audio.",
        size=11, colour=MUTED, italic=True, spacing=1.2)

    # ---------------------------------------------------------- 21  replay
    s = slide_full(prs, "Herath H.M.M.P.B", "Detecting a replayed recording",
                   21, 3,
                   lede="ASVspoof 2017 V2 physical replay attacks, 1,040 "
                        "held-out development files. Same CNN throughout; "
                        "only the front-end changes.")
    hero_number(s, M, 2.32, "4.90%", "EER with an inverted-Mel front-end",
                accent=WIN,
                sub="down from 10.00% with a standard Log-Mel front-end on "
                    "the identical network")
    tile(s, M, 4.78, 1.90, 1.10, "0.955", "replay F1", accent=ORANGE)
    tile(s, 2.78, 4.78, 1.90, 1.10, "95.1%", "accuracy", accent=INK)

    column_chart(s, 5.20, 2.18, 3.40, 3.60,
                 ["Log-Mel", "Inverted Mel"],
                 [("EER %", (10.00, 4.90), NEUTRAL)], gap=95,
                 point_colours=[NEUTRAL, WIN])
    note(s, 8.95, 2.18, 3.63, 3.60, "Why the front-end decides this",
         "A Mel filterbank spends most of its resolution on low frequencies, "
         "where the speech energy is.\n\nInverted Mel spends it on high "
         "frequencies instead, which is exactly where a playback loudspeaker "
         "and a second microphone leave their fingerprint.\n\nThe network, "
         "the data and the training schedule were unchanged, so the gain is "
         "attributable to the front-end alone.", tone="wash")

    # ---------------------------------------------------------- 22  replay lim
    s = slide_full(prs, "Herath H.M.M.P.B",
                   "Where replay detection stops working", 22, 3,
                   lede="Every row is a zero-shot run: trained on one corpus, "
                        "scored on another with no retraining.")
    red = {"c": FAIL, "b": True, "a": PP_ALIGN.RIGHT}
    table(s, M, 2.16, 5.75,
          [["Trained on \u2192 scored on",
            {"t": "EER", "a": PP_ALIGN.RIGHT}],
           ["ASVspoof 2017 \u2192 2019 PA", dict(red, t="50.47%")],
           ["2019 PA \u2192 ASVspoof 2017", dict(red, t="39.24%")],
           ["2017 + PA \u2192 2019 LA (TTS/VC)", dict(red, t="41.12%")],
           ["AASIST off-the-shelf \u2192 2017", dict(red, t="64.32%")]],
          col_w=[4.05, 1.70], row_h=0.44, head_h=0.42, size=11.5,
          head_fill=ORANGE_TX)
    W(textbox(s, M, 4.54, 5.75, 0.3)).eyebrow(
        "Mixed-corpus training, three front-ends")
    right = {"a": PP_ALIGN.RIGHT}
    win = {"c": WIN, "b": True, "a": PP_ALIGN.RIGHT}
    table(s, M, 4.82, 5.75,
          [["Front-end", {"t": "2017", "a": PP_ALIGN.RIGHT},
            {"t": "PA 2019", "a": PP_ALIGN.RIGHT}],
           ["Log-Mel", dict(right, t="21.17%"), dict(right, t="5.99%")],
           ["Inverted Mel", dict(right, t="10.12%"), dict(right, t="9.66%")],
           ["LFCC", dict(win, t="9.24%"), dict(win, t="8.98%")]],
          col_w=[2.55, 1.60, 1.60], row_h=0.44, head_h=0.40, size=11.5,
          head_fill=INK)

    note(s, 6.83, 2.16, 5.75, 1.40, "Fix 1 \u2014 train on the target corpus",
         "Training the same network directly on ASVspoof 2019 PA reaches "
         "7.96% EER over 15,434 development files, at an F1 of 0.949.",
         accent=WIN)
    note(s, 6.83, 3.62, 5.75, 1.40, "Fix 2 \u2014 mix the corpora",
         "One corpus-balanced model holds on both instead of collapsing on "
         "the unseen one: 9.18% on 2017 and 11.00% on PA.", accent=WIN)
    note(s, 6.83, 5.08, 5.75, 1.46,
         "The consequence \u2014 why AASIST was needed",
         "At 41.12% on logical access, a replay detector is blind to "
         "synthesised and converted speech. A different countermeasure has "
         "to carry that attack, which is the next slide.", tone="solid")

    # ---------------------------------------------------------- 23  sasv
    s = slide_split(
        prs, "Herath H.M.M.P.B", "Anti-spoofing\nand the joint\nSASV metric",
        "A recording of the real user passes speaker verification. That is "
        "the hole the countermeasure has to close.", 23, 3)
    wr = W(textbox(s, MAIN_X, 1.00, MAIN_W, 1.9))
    for t in ("Evaluated on ASVspoof 2019 LA with the official SASV 2022 "
              "trial lists",
              "MUSAN noise added on the test side only, at clean / 15 / 10 / "
              "5 / 0 dB",
              "Tuned on dev, locked the evaluation set once, with no further "
              "tuning"):
        wr.bullet(t, size=12.5, accent=ORANGE, space_after=11)
    table(s, MAIN_X, 3.06, MAIN_W,
          [["System", "What it combines"],
           ["B0", "ECAPA speaker score only"],
           ["B1", "ECAPA + AASIST countermeasure (\u03b1 = 0.30)"],
           ["P2 / P3", "SNR-gated and always-on enhancement"]],
          col_w=[1.40, 5.88], row_h=0.50, head_h=0.44, size=11.5,
          head_fill=ORANGE_TX)
    cx = MAIN_X
    for name, desc, accent in (("SV-EER", "target vs non-target", ORANGE),
                               ("SPF-EER", "target vs spoof", ORANGE),
                               ("SASV-EER", "both at once", ORANGE)):
        panel(s, cx, 5.26, 2.29, 1.34)
        wr = W(textbox(s, cx + 0.26, 5.52, 1.80, 0.9))
        wr.eyebrow(name, colour=accent, size=10, y_after=6)
        wr.body(desc, size=10.5, colour=MUTED, spacing=1.14)
        cx += 2.50

    # ---------------------------------------------------------- 24  sasv res
    s = slide_full(prs, "Herath H.M.M.P.B", "One clear win, one honest "
                   "failure", 24, 3,
                   lede="Locked evaluation on clean audio, then the same "
                        "systems under test-side noise.")
    hero_number(s, M, 2.32, "0.83%", "SASV-EER with AASIST fusion",
                accent=WIN,
                sub="down from 20.67% for ECAPA alone, and better than the "
                    "published 1.71% baseline")

    bar_chart(s, 4.55, 2.18, 4.05, 2.55,
              ["ECAPA only", "+ LFCC", "+ AASIST"],
              (20.67, 7.13, 0.83), [FAIL, NEUTRAL, WIN])

    note(s, 8.86, 2.18, 3.72, 2.55, "Result 1 \u2014 the countermeasure is "
         "essential",
         "A strong speaker encoder on its own fails against spoofed speech, "
         "at 27% spoof error. Fusing AASIST with a tuned weight closes it.",
         accent=WIN)

    table(s, 4.55, 5.00, 4.05,
          [["System @ 15 dB", "SASV-EER"],
           ["B0 raw ECAPA", {"t": "16.91%", "b": True}],
           ["P2 gated enhance", {"t": "16.93%", "c": FAIL, "b": True}],
           ["P3 always enhance", {"t": "23.11%", "c": FAIL, "b": True}]],
          col_w=[2.45, 1.60], row_h=0.42, head_h=0.40, size=11,
          head_fill=ORANGE_TX)
    note(s, 8.86, 5.00, 3.72, 1.68, "Result 2 \u2014 an honest negative",
         "Gating the enhancer by estimated SNR did not beat raw ECAPA on "
         "SASV, and always enhancing was clearly worse.", accent=FAIL)

    # ---------------------------------------------------------- 25  fit
    s = slide_full(prs, "Research track", "How the three tracks fit together",
                   25, 3)
    parts = [("Identity", "Is this the same speaker?",
              "ECAPA and TitaNet embeddings, VAD gating and a pgvector "
              "template store.", "Muditha", ORANGE),
             ("Robustness", "Can we still verify in noise?",
              "A task-tuned Wave-U-Net plus learned fusion of raw and "
              "enhanced embeddings.", "Yasith", ORANGE),
             ("Security", "Is the audio real?",
              "A replay CNN and the AASIST countermeasure fused into a joint "
              "SASV decision.", "H.M.M.P.B", ORANGE)]
    cx = M
    for label, question, body, who, accent in parts:
        panel(s, cx, 2.16, 3.78, 2.30)
        pill(s, cx, 2.16, 3.78, 0.07, fill=accent)
        wr = W(textbox(s, cx + 0.32, 2.32, 3.14, 2.00,
                       anchor=MSO_ANCHOR.MIDDLE))
        wr.eyebrow(label, colour=accent, size=9.5, y_after=7)
        wr.body(question, size=14.5, colour=INK, bold=True, space_after=7,
                font=DISPLAY, spacing=1.08)
        wr.body(body, size=10.5, colour=MUTED, space_after=8, spacing=1.2)
        wr.eyebrow(who, colour=accent, size=9)
        cx += 4.03
    note(s, M, 4.72, CW, 1.72, "Reconciling the two enhancement findings",
         "Learned fusion improved speaker verification under MUSAN noise on "
         "VoxCeleb1, while a simple SNR-gated enhancer failed on the SASV "
         "task on ASVspoof LA. These are different protocols, different "
         "attacks and different decision rules, so both results stand: "
         "enhancement pays off only when the blend is learned for the exact "
         "task.", tone="solid")

    # ---------------------------------------------------------- 26  demo
    s = slide_split(
        prs, "Demonstration", "The system\nin use",
        "Enroll once, verify in a second, and watch a replayed recording "
        "get turned away.", 26, 4)
    picture(s, "v3_demo.png", MAIN_X, 1.02, MAIN_W, radius=R_CARD)
    wr = W(textbox(s, MAIN_X, 5.22, MAIN_W, 1.4))
    for t, accent in (("Sign in, then enroll with a few short recordings",
                       ORANGE),
                      ("Verify with a fresh recording and read the score",
                       ORANGE),
                      ("Replay a recording into the mic and watch it be "
                       "rejected", ORANGE),
                      ("Push to main and see the cloud stack rebuild itself",
                       ORANGE)):
        wr.bullet(t, size=11.5, accent=accent, space_after=7)
    note(s, M, 5.30, RAIL_W, 1.30, "If the room audio is bad",
         "Play the recorded walkthrough, then show the green deployment run "
         "and the live service.", tone="wash")

    # ---------------------------------------------------------- 27  limits
    s = slide_full(prs, "Reflection", "Limitations and next steps", 27, 4)
    W(textbox(s, M, 2.04, 5.75, 0.3)).eyebrow("What we would not claim",
                                              colour=FAIL)
    W(textbox(s, 6.83, 2.04, 5.75, 0.3)).eyebrow("What we would do next")
    limits = ["EER values come from different protocols and are not directly "
              "comparable across tracks",
              "Noise was applied mainly on the test side while enrollment "
              "stayed clean",
              "Replay detection does not transfer across corpora, so new "
              "microphones and rooms need their own data",
              "Parts of the noisy-SASV grid are smoke-scale rather than full "
              "runs"]
    nexts = ["Apply the learned fusion to the SASV task instead of a fixed "
             "SNR gate",
             "Complete the countermeasure-under-noise matrix and lock it once",
             "Calibrate thresholds per SNR band so the decision stays stable",
             "Finish the TitaNet comparison and collect real device "
             "recordings"]
    for i, (lt, nx) in enumerate(zip(limits, nexts)):
        y = 2.44 + i * 0.98
        panel(s, M, y, 5.75, 0.86)
        pill(s, M + 0.24, y + 0.21, 0.05, 0.44, fill=FAIL)
        W(textbox(s, M + 0.48, y, 5.05, 0.86,
                  anchor=MSO_ANCHOR.MIDDLE)).body(
            lt, size=11, colour=INK, spacing=1.2)
        panel(s, 6.83, y, 5.75, 0.86)
        pill(s, 7.07, y + 0.21, 0.05, 0.44, fill=WIN)
        W(textbox(s, 7.31, y, 5.05, 0.86,
                  anchor=MSO_ANCHOR.MIDDLE)).body(
            nx, size=11, colour=INK, spacing=1.2)
    W(textbox(s, M, 6.42, CW, 0.4)).rich([
        ("REPORTING DISCIPLINE     ", 10, INK, True, LABEL),
        ("every number in this deck comes from a run we can reproduce, "
         "including the experiments that did not work.", 11.5, MUTED, False,
         BODY)])

    # ---------------------------------------------------------- 28  close
    s = blank(prs)
    set_bg(s, DARK)
    picture(s, "v3_closing.png", 0.0, 0.0, SW)
    W(textbox(s, 0.95, 0.72, 7.0, 0.3)).eyebrow("In summary", colour=D_ORANGE,
                                                size=11)
    W(textbox(s, 0.95, 1.08, 9.3, 0.9)).title(
        "Designed, measured, and deployed", size=38, colour=D_INK)
    closers = [("0.83%", "joint SASV error, clean audio", D_WIN),
               ("8.79%", "speaker error across all noise", D_WIN),
               ("0.54%", "VoxCeleb error with gating", D_WIN)]
    cx = 0.95
    for value, label, accent in closers:
        tile(s, cx, 2.24, 3.50, 1.10, value, label, accent=accent, dark=True)
        cx += 3.70
    W(textbox(s, 0.95, 4.86, 11.0, 0.8)).body(
        "We built the running product, measured every track on a standard "
        "protocol, reported the experiments that failed, and automated the "
        "deployment.", size=13, colour=D_ORANGE, spacing=1.3)
    W(textbox(s, 0.95, 5.94, 6.5, 0.6)).title("Thank you \u2014 questions?",
                                              size=26, colour=D_INK)
    W(textbox(s, 7.6, 6.14, 4.78, 0.4)).body(
        "Muditha Herath  \u00b7  Yasith Hewarathna  \u00b7  Herath H.M.M.P.B",
        size=10.5, colour=D_MUTED, align=PP_ALIGN.RIGHT)

    # ---------------------------------------------------------- notes + nav
    notes = [
        "Open with the one-line pitch: a voice login that checks who you are, whether the voice is real, and whether it survives noise. (30 s)",
        "Divider. Say one sentence and move on. (10 s)",
        "Frame the pain, then the two threats. This slide earns the rest of the talk. (1 min)",
        "Define the vocabulary once so the numbers later land. Stress that SASV is the honest metric. (45 s)",
        "The value slide. One line per card, do not read them out verbatim. (1 min)",
        "Divider. (10 s)",
        "Walk left to right, then down, then the ML panel. Point out the spoof check happens before identity matching. (2 min)",
        "Summarise only: capture, gate, check, embed, fuse, decide. (1 min)",
        "Do not read the tables. Web stack left, speech models right, one shared contract. (45 s)",
        "Pick two decisions, ideally 'never enhance blindly' and the spoof gate, because they set up the research. (1 min)",
        "Divider. (10 s)",
        "Switch from engineering to product. Name one concrete buyer and workflow. (1 min)",
        "Tie each KPI to a number we measured. End on the responsible-use card. (1 min)",
        "Divider. (10 s)",
        "Introduce the three tracks. The right-hand cards are clickable if someone wants to jump. (45 s)",
        "Muditha's scope. This is the part that actually runs in production. (1 min)",
        "Give the number first, then the honest reading: small EER gain, real robustness gain. (1.5 min)",
        "Yasith's core idea: do not choose between raw and enhanced, learn the blend. (1.5 min)",
        "Five phases, quickly. Land on the negative result at the bottom. (1.5 min)",
        "The strongest research result. All-noisy error 12.58 down to 8.79 with almost no clean cost. (1.5 min)",
        "Start the security track with the literal threat: a recording played back at the microphone. The headline is that the front-end, not the network, decided this result. Inverted Mel puts resolution on high frequencies where the playback speaker and the second microphone leave artefacts, and that alone took 10.00% down to 4.90% on the same CNN. If asked: ASVspoof 2017 V2, 1,040 held-out development files, speaker-disjoint from training. (1.5 min)",
        "The honest boundary slide. Say it plainly: a replay detector trained on one corpus is worthless on another, 50.47% and 39.24% are both chance-level. Off-the-shelf AASIST is no better zero-shot at 64.32%, and a frozen wav2vec2 with a trained head only reached 37.32%, so a small task-trained CNN beats the big pretrained models here. Two fixes work: train on the target corpus (7.96% on PA) or mix corpora (9.18% and 11.00%). Land on the last card: 41.12% on logical access means replay detection cannot see text-to-speech, which is why the next slide brings in AASIST. (2 min)",
        "Set up the security track's joint metric. A recording of the real user passes speaker verification. (1 min)",
        "Two results. Deliver the failure confidently: it is a finding, not a mistake. (2 min)",
        "Pre-empt the obvious question about why enhancement worked once and failed once. (1 min)",
        "Run the demo. If the room audio is unreliable, go straight to the recording. (2-3 min)",
        "Say the limitations before the panel does, each with a next step. (1 min)",
        "Close on the three numbers and one sentence, then stop and take questions. (45 s)",
    ]
    for slide, text in zip(prs.slides, notes):
        slide.notes_slide.notes_text_frame.text = text

    sl = prs.slides
    for i, target in zip(range(3), (15, 17, 20)):
        nav[i].click_action.target_slide = sl[target]

    prs.save(OUT)
    print(f"Saved: {OUT}  ({len(sl._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
