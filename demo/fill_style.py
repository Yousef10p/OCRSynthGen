"""Text and style for filling bank templates: our sample Arabic text and a random typography per page.

Shared by demo/render_demo.py and synthdoc_adapter/, so it needs only Pillow (no OCRSmith install);
the fonts themselves are read from OCRSmith's font folder.
"""
import html
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
OCRSMITH = ROOT.parent / "ocrsmith" / "assets"
INKS = [(0, 0, 0), (20, 20, 20), (30, 30, 45), (10, 25, 60), (45, 35, 30), (60, 10, 10)]
WEIGHTS = {"thin": 100, "extralight": 200, "light": 300, "regular": 400, "medium": 500, "semibold": 600,
           "bold": 700, "extrabold": 800, "black": 900}

# Our own neutral sentences; repetition across pages is fine for a demo.
SENTENCES = [
    "تعمل المؤسسة على تطوير خدماتها الرقمية لتسهيل وصول المواطنين إلى المعلومات.",
    "شهد العام الماضي نموا ملحوظا في عدد المشاريع الصغيرة والمتوسطة.",
    "يهدف هذا التقرير إلى عرض أهم النتائج التي توصل إليها فريق العمل.",
    "تقع المدينة على ضفاف النهر وتشتهر بأسواقها القديمة وحدائقها الواسعة.",
    "يرجى تعبئة النموذج بخط واضح وإرفاق المستندات المطلوبة قبل الموعد المحدد.",
    "ارتفعت نسبة المشاركة في البرنامج التدريبي إلى أكثر من سبعين في المئة.",
    "تتضمن الخطة الجديدة تحسين البنية التحتية وتوسيع شبكة النقل العام.",
    "أكدت اللجنة أهمية التعاون بين الجامعات والقطاع الخاص في مجال البحث العلمي.",
    "يقدم المركز دورات مجانية في اللغة العربية والحاسوب والمهارات المهنية.",
    "تم تحديث قاعدة البيانات لتشمل جميع السجلات الصادرة خلال الربع الأخير.",
    "تعتمد الدراسة على بيانات ميدانية جمعت من عدة مناطق مختلفة.",
    "يسر الإدارة أن تعلن عن افتتاح الفرع الجديد في الأسبوع القادم.",
    "يجب مراجعة الشروط والأحكام بعناية قبل التوقيع على العقد.",
    "حققت الشركة أرباحا صافية تجاوزت توقعات المحللين لهذا العام.",
    "تساهم القراءة اليومية في تنمية مهارات التفكير والتعبير لدى الطلاب.",
    "أوصى الاجتماع بإعداد دليل شامل للإجراءات الإدارية المعتمدة.",
    "يمكن التواصل مع خدمة العملاء عبر الهاتف أو البريد الإلكتروني.",
    "تشير المؤشرات الأولية إلى تحسن الأداء في معظم القطاعات الاقتصادية.",
]
WORDS = [w.strip(".،") for s in SENTENCES for w in s.split()]
# English pages (the pipeline fills a share of templates in English, left to right)
SENTENCES_EN = [
    "The institution is developing its digital services to make information easier to reach.",
    "Last year saw clear growth in the number of small and medium-sized businesses.",
    "This report presents the main findings reached by the working group.",
    "The city lies on the bank of the river and is known for its old markets and wide gardens.",
    "Please fill in the form clearly and attach the required documents before the deadline.",
    "Participation in the training programme rose to more than seventy percent.",
    "The new plan includes better infrastructure and a wider public transport network.",
    "The committee stressed the value of cooperation between universities and industry.",
    "The centre offers free courses in languages, computing and professional skills.",
    "The database has been updated to include all records issued in the last quarter.",
    "The study is based on field data collected from several different regions.",
    "The management is pleased to announce the opening of a new branch next week.",
    "The terms and conditions should be reviewed carefully before signing the contract.",
    "The company reported a net profit that exceeded analyst expectations this year.",
    "Daily reading helps students develop their thinking and writing skills.",
    "The meeting recommended preparing a complete guide to the approved procedures.",
    "Customer service can be reached by telephone or by electronic mail.",
    "Early indicators point to better performance in most sectors of the economy.",
]
WORDS_EN = [w.strip(".,") for s in SENTENCES_EN for w in s.split()]
# Windows fonts with Latin glyphs, used by name (no @font-face needed): Chromium finds them installed
LATIN_FONTS = ["Georgia", "Times New Roman", "Cambria", "Constantia", "Book Antiqua", "Palatino Linotype",
               "Arial", "Calibri", "Segoe UI", "Verdana", "Tahoma", "Candara", "Corbel", "Trebuchet MS"]
ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


# -- text ------------------------------------------------------------------------------------------

def prose(rng, n, sentences=SENTENCES):
    """n words of running text, with sentence ends and the odd comma."""
    out = []
    while len(out) < n:
        out += rng.choice(sentences).split()
    words = out[:n]
    if not words[-1].endswith("."):
        words[-1] = words[-1].rstrip("،") + "."
    return " ".join(words)


def phrase(rng, n, words=WORDS):
    start = rng.randrange(len(words))
    return " ".join(words[(start + i) % len(words)] for i in range(n))


def number(rng):
    return rng.choice([str(rng.randint(1, 999)), f"{rng.randint(1, 99)}.{rng.randint(0, 99):02d}",
                       f"{rng.randint(1, 99)}%", f"{rng.randint(1000, 99999):,}"])


def field(rng):
    return rng.choice([f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2025)}",
                       f"05{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
                       f"{rng.randint(100, 99999):,}.{rng.randint(0, 99):02d}", str(rng.randint(10000, 999999))])


def fill(slot, rng, digits, lang="ar"):
    """HTML-escaped text for one slot: right kind of text for its role, random length in its word range.

    lang="en" fills it with English instead of Arabic."""
    sentences, words = (SENTENCES_EN, WORDS_EN) if lang == "en" else (SENTENCES, WORDS)
    lo, hi = slot["words"]
    n = rng.randint(lo, max(lo, hi))
    role = slot["role"]
    if role == "page_number":
        text = str(rng.randint(1, 120))
    elif role == "cell" and rng.random() < 0.55:
        text = number(rng)
    elif role == "field_value" and rng.random() < 0.6:
        text = field(rng)
    elif role == "label":
        text = phrase(rng, n, words) + ":"
    elif role in ("paragraph", "quote", "footnote", "sidebar", "list_item", "caption"):
        text = prose(rng, n, sentences)
    else:
        text = phrase(rng, n, words)
    return html.escape(text.translate(ARABIC_DIGITS) if digits == "arabic-indic" else text)


# -- style -----------------------------------------------------------------------------------------

def font_families():
    """Arabic families in OCRSmith's font folder: {family: {(weight, italic): file}}."""
    families = {}
    for path in sorted((OCRSMITH / "fonts").glob("*.[ot]tf")):
        family, style = ImageFont.truetype(str(path), 12).getname()
        if "Mono" in family:  # Latin only
            continue
        key = style.lower().replace(" ", "")
        weight = next((w for k, w in sorted(WEIGHTS.items(), key=lambda kv: -len(kv[0])) if k in key), 400)
        families.setdefault(family, {})[(weight, "italic" in key)] = path
    return families


def pick_weight(rng, faces, wanted):
    weights = sorted({w for w, italic in faces if not italic})
    return min(weights, key=lambda w: abs(w - rng.choice(wanted)))


def style(rng, families, transparent=True):
    """Random typography for one page, written as CSS for the template's {{ style }} hook.

    transparent=True clears the page background, so the render can be laid on paper afterwards."""
    body_family = rng.choice(sorted(families))
    head_family = body_family if rng.random() < 0.7 else rng.choice(sorted(families))
    body_w = pick_weight(rng, families[body_family], [300, 400, 400, 500])
    head_w = pick_weight(rng, families[head_family], [600, 700, 700, 800, 900])
    s = page_style(rng, body_family, head_family, body_w, head_w)
    faces = "".join(
        f'@font-face {{ font-family: "{fam}"; src: url("{path.as_uri()}"); font-weight: {w};'
        f' font-style: {"italic" if it else "normal"}; }}' + "\n"
        for fam in sorted({body_family, head_family}) for (w, it), path in families[fam].items())
    return s, faces + style_css(s, transparent)


def style_latin(rng, transparent=True):
    """The same random typography for an English page, from installed Latin fonts."""
    body_family = rng.choice(LATIN_FONTS)
    head_family = body_family if rng.random() < 0.7 else rng.choice(LATIN_FONTS)
    s = page_style(rng, body_family, head_family, 400, 700, "western")
    # Arabic letter numbering in lists means nothing on an English page
    return s, style_css(s, transparent) + "ol { list-style-type: decimal !important; }\n"


def page_style(rng, body_family, head_family, body_w, head_w, digits=None):
    """Size, spacing, alignment, margins, ink and digits (picked at random unless given), in a fixed draw order."""
    ink = rng.choice(INKS)
    return {"font": body_family, "heading_font": head_family, "body_weight": body_w, "heading_weight": head_w,
            "size_pt": round(rng.uniform(9, 14), 1), "line_height": round(rng.uniform(1.35, 2.0), 2),
            "align": rng.choice(["start", "justify", "justify"]), "margin_mm": rng.randint(10, 28),
            "ink": f"rgb{ink}", "digits": digits or rng.choice(["western", "western", "arabic-indic"])}


def style_css(s, transparent):
    clear = " background: transparent !important;" if transparent else ""
    return (f':root {{ --font-body: "{s["font"]}", serif; --font-heading: "{s["heading_font"]}", serif;'
            f' --base-size: {s["size_pt"]}pt; --line-height: {s["line_height"]}; --align: {s["align"]};'
            f' --page-margin: {s["margin_mm"]}mm; }}\n'
            + ("html, body {" + clear + " }\n" if transparent else "")
            + f".page {{ color: {s['ink']}; font-weight: {s['body_weight']};{clear} }}\n"
            f"h1, h2, h3, h4, th {{ font-weight: {s['heading_weight']}; }}\n")
