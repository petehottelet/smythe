from PIL import Image, ImageDraw, ImageFont
import os

REPO = "/sessions/awesome-charming-cray/mnt/smythe"
OUT = os.path.join(REPO, "assets", "osiris_ads")
os.makedirs(OUT, exist_ok=True)

OFFWHITE = (250, 246, 236)
BLACK = (26, 26, 24)
AMBER = (224, 154, 36)
AMBER_DARK = (154, 123, 45)

LATO = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
LATO_LIGHT = "/usr/share/fonts/truetype/lato/Lato-Light.ttf"
LATO_BOLD = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"

logo = Image.open(os.path.join(REPO, "00_project_files", "osiris_light.png")).convert("RGBA")
icon = logo.crop((0, 0, logo.width, 748))
wordmark = logo.crop((0, 800, logo.width, 964))

def _bbox_crop(im):
    return im.crop(im.getchannel("A").getbbox())

icon = _bbox_crop(icon)
wordmark = _bbox_crop(wordmark)
full = _bbox_crop(logo)
sun = _bbox_crop(icon.crop((0, 0, icon.width, round(icon.height * 0.40))))

def scaled(im, h=None, w=None):
    if h is not None:
        w = round(im.width * h / im.height)
    else:
        h = round(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)

def font(path, size):
    return ImageFont.truetype(path, size)

def text_w(d, s, f):
    b = d.textbbox((0, 0), s, font=f)
    return b[2] - b[0]

TAGLINE = "POWER FROM THE SUN"
CTA = "LEARN MORE"

def canvas(w, h):
    im = Image.new("RGBA", (w, h), OFFWHITE + (255,))
    return im, ImageDraw.Draw(im)

def frame(d, w, h, inset=None):
    if inset is None:
        inset = max(3, round(min(w, h) * 0.028))
    d.rectangle([inset, inset, w - 1 - inset, h - 1 - inset], outline=AMBER_DARK, width=1)
    return inset

def cta_pill(d, im, cx, cy, fsize):
    f = font(LATO_BOLD, fsize)
    tw = text_w(d, CTA, f)
    pad_x = round(fsize * 0.9)
    pad_y = round(fsize * 0.55)
    w2, h2 = tw + 2 * pad_x, fsize + 2 * pad_y
    box = [cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2]
    d.rounded_rectangle(box, radius=h2 // 2, fill=BLACK)
    d.text((cx - tw / 2, cy - fsize / 2 - fsize * 0.12), CTA, font=f, fill=AMBER)

def paste(im, art, x, y):
    im.alpha_composite(art, (round(x), round(y)))

def horizontal(w, h, name, show_tagline=True):
    im, d = canvas(w, h)
    inset = frame(d, w, h)
    pad = inset + max(6, round(h * 0.10))
    ic = scaled(icon, h=round(h * 0.72))
    ix = pad + round(h * 0.06)
    paste(im, ic, ix, (h - ic.height) / 2 - h * 0.02)
    wm_h = round(h * (0.30 if show_tagline else 0.34))
    wm = scaled(wordmark, h=wm_h)
    wx = ix + ic.width + round(h * 0.22)
    max_wm_w = w - pad - wx - round(h * 0.15)
    if wm.width > max_wm_w:
        wm = scaled(wordmark, w=max_wm_w)
    if show_tagline:
        block_h = wm.height + round(h * 0.10) + round(h * 0.17)
        wy = (h - block_h) / 2
        paste(im, wm, wx, wy)
        f = font(LATO, round(h * 0.155))
        d.text((wx + 2, wy + wm.height + round(h * 0.115)), TAGLINE, font=f, fill=AMBER_DARK)
        right_edge = wx + max(wm.width, text_w(d, TAGLINE, f))
    else:
        wy = (h - wm.height) / 2
        paste(im, wm, wx, wy)
        right_edge = wx + wm.width
    fsize = max(10, round(h * 0.20))
    fb = font(LATO_BOLD, fsize)
    tw = text_w(d, CTA, fb)
    pill_w = tw + round(fsize * 1.8)
    cx = w - pad - pill_w // 2 - round(h * 0.06)
    if cx - pill_w // 2 > right_edge + 10:
        cta_pill(d, im, cx, h // 2, fsize)
    im.convert("RGB").save(os.path.join(OUT, name), quality=95)

def rectangle(w, h, name):
    im, d = canvas(w, h)
    frame(d, w, h)
    lg = scaled(full, h=round(h * 0.52))
    if lg.width > w * 0.72:
        lg = scaled(full, w=round(w * 0.72))
    top = round(h * 0.10)
    paste(im, lg, (w - lg.width) / 2, top)
    y = top + lg.height + round(h * 0.055)
    fsize = max(11, round(w * 0.042))
    f = font(LATO, fsize)
    tw = text_w(d, TAGLINE, f)
    d.text(((w - tw) / 2, y), TAGLINE, font=f, fill=AMBER_DARK)
    cta_pill(d, im, w // 2, round(h * 0.855), max(12, round(w * 0.048)))
    im.convert("RGB").save(os.path.join(OUT, name), quality=95)

def vertical(w, h, name):
    im, d = canvas(w, h)
    inset = frame(d, w, h)
    lg = scaled(full, w=round(w * 0.76))
    top = round(h * 0.06)
    paste(im, lg, (w - lg.width) / 2, top)
    y = top + lg.height + round(h * 0.035)
    fsize = max(11, round(w * 0.062))
    f = font(LATO, fsize)
    if text_w(d, TAGLINE, f) < w * 0.84:
        tw = text_w(d, TAGLINE, f)
        d.text(((w - tw) / 2, y), TAGLINE, font=f, fill=AMBER_DARK)
        y += round(fsize * 1.6)
    else:
        for wd in ["POWER", "FROM", "THE SUN"]:
            tw = text_w(d, wd, f)
            d.text(((w - tw) / 2, y), wd, font=f, fill=AMBER_DARK)
            y += round(fsize * 1.45)
    y += round(h * 0.008)
    d.line([w * 0.32, y, w * 0.68, y], fill=AMBER, width=2)
    y += round(h * 0.028)
    f2 = font(LATO_LIGHT, max(10, round(w * 0.052)))
    for line in ["PORTABLE", "SOLAR CHARGER"]:
        tw = text_w(d, line, f2)
        d.text(((w - tw) / 2, y), line, font=f2, fill=BLACK)
        y += round(f2.size * 1.4)
    cta_pill(d, im, w // 2, max(round(y + h * 0.06), round(h * 0.745)), max(13, round(w * 0.075)))
    sn = scaled(sun, w=round(w * 0.58))
    arc = sn.crop((0, 0, sn.width, round(sn.height * 0.62)))
    paste(im, arc, (w - arc.width) / 2, h - inset - arc.height - 1)
    d.line([inset + w * 0.06, h - inset - 1, w - inset - w * 0.06, h - inset - 1], fill=AMBER_DARK, width=1)
    im.convert("RGB").save(os.path.join(OUT, name), quality=95)

horizontal(728, 90, "osiris_728x90_large_leaderboard.png")
horizontal(320, 100, "osiris_320x100_large_mobile_banner.png")
horizontal(320, 50, "osiris_320x50_mobile_leaderboard.png", show_tagline=False)
rectangle(336, 280, "osiris_336x280_large_rectangle.png")
rectangle(300, 250, "osiris_300x250_inline_rectangle.png")
vertical(300, 600, "osiris_300x600_half_page.png")
vertical(160, 600, "osiris_160x600_wide_skyscraper.png")

PAD = 24
sheet = Image.new("RGB", (728 + 160 + 300 + PAD * 4, 600 + PAD * 2 + 40), (233, 226, 210))
x = PAD
for name in ["osiris_160x600_wide_skyscraper.png", "osiris_300x600_half_page.png"]:
    im = Image.open(os.path.join(OUT, name))
    sheet.paste(im, (x, PAD))
    x += im.width + PAD
col_x = x
y = PAD
for name in ["osiris_728x90_large_leaderboard.png", "osiris_336x280_large_rectangle.png", "osiris_300x250_inline_rectangle.png"]:
    im = Image.open(os.path.join(OUT, name))
    sheet.paste(im, (col_x, y))
    y += im.height + PAD
x2 = col_x + 336 + PAD
y2 = PAD + 90 + PAD
for name in ["osiris_320x100_large_mobile_banner.png", "osiris_320x50_mobile_leaderboard.png"]:
    im = Image.open(os.path.join(OUT, name))
    sheet.paste(im, (x2, y2))
    y2 += im.height + PAD
sheet.save(os.path.join(OUT, "contact_sheet.png"))
print("done")
