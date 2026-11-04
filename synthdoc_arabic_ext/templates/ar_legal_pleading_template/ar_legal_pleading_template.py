"""
Arabic statement of claim (صحيفة دعوى), from the legal-pleading section of
SynthDoc's ``test_showcase`` template: court caption box, attorney block and
numbered paragraphs under section headings.
"""
from datetime import timedelta
from typing import Any, Dict

from synthdoc.templates.test_showcase_template.test_showcase_template import TestShowcaseTemplate
from synthdoc_arabic.base import ArabicTemplateMixin

COURTS = ["المحكمة التجارية", "المحكمة العامة", "المحكمة العمالية"]
CIRCUITS = ["الدائرة الأولى", "الدائرة الثانية", "الدائرة الثالثة", "الدائرة الخامسة"]
CLAIMS = ["المطالبة بمبلغ مالي", "فسخ العقد", "التعويض عن الضرر", "الإخلال بالالتزامات التعاقدية",
          "رد الدفعة المقدمة"]
SERVICES = ["أعمال توريد وتركيب أنظمة التكييف المركزي", "تصميم وتنفيذ نظام إلكتروني لإدارة المخزون",
            "أعمال الترميم والتشطيب الداخلي لمبنى إداري", "توريد معدات طبية لعيادة متخصصة",
            "تنفيذ حملة تسويقية متكاملة لمدة ستة أشهر"]

PARTIES = [
    "المدعي {P} مقيم في مدينة {city}، ويتمتع بالأهلية الكاملة للتقاضي، ويمثله في هذه الدعوى "
    "المحامي {A} بموجب وكالة سارية المفعول.",
    "المدعى عليها {D} منشأة مسجلة بموجب السجل التجاري رقم {cr}، ويقع مقرها الرئيس في مدينة {city2}.",
    "تختص المحكمة الموقرة بنظر هذه الدعوى نوعياً ومكانياً، لكون العقد محل النزاع قد أُبرم ونُفذ "
    "في دائرة اختصاصها.",
]
FACTS = [
    "بتاريخ {date1} أبرم المدعي مع المدعى عليها العقد رقم {contract} الذي تلتزم بموجبه المدعى عليها "
    "بتنفيذ {service} خلال مدة لا تتجاوز {period} يوماً من تاريخ التوقيع.",
    "سدد المدعي دفعة مقدمة قدرها {amount1} وفقاً لما نص عليه البند الرابع من العقد، وتسلمت المدعى عليها "
    "المبلغ بموجب إيصال رسمي.",
    "انقضت المدة المتفق عليها دون أن تفي المدعى عليها بالتزاماتها التعاقدية، ولم تقدم أي مبرر نظامي "
    "لهذا التأخير.",
    "وجّه المدعي إنذاراً كتابياً إلى المدعى عليها بتاريخ {date2} يطالبها فيه بتنفيذ التزاماتها أو "
    "إعادة المبلغ المدفوع، إلا أنها لم تستجب لذلك.",
    "ترتب على إخلال المدعى عليها بالتزاماتها أضرار مادية لحقت بالمدعي، تمثلت في تعطل أعماله وتكبده "
    "نفقات إضافية بلغت {amount2}.",
    "حاول المدعي تسوية النزاع ودياً عبر عدة مراسلات واجتماعات، غير أن تلك المحاولات لم تُسفر عن أي نتيجة.",
]
GROUNDS = [
    "تنص القواعد العامة للالتزامات على أن العقد شريعة المتعاقدين، فلا يجوز لأحد طرفيه نقضه أو تعديله "
    "إلا باتفاق الطرفين.",
    "يحق للمتعاقد عند إخلال الطرف الآخر بالتزامه أن يطالب بتنفيذ العقد أو بفسخه مع التعويض في الحالتين "
    "إن كان له مقتضى.",
    "إن كل خطأ سبب ضرراً للغير يُلزم من ارتكبه بالتعويض، وهو ما تحقق في هذه الدعوى بأركانه الثلاثة.",
]
REQUESTS = [
    "إلزام المدعى عليها بأن تدفع للمدعي مبلغاً قدره {total}.",
    "فسخ العقد رقم {contract} واعتباره كأن لم يكن.",
    "إلزام المدعى عليها بتعويض المدعي عن الأضرار المادية التي لحقت به.",
    "إلزام المدعى عليها بأتعاب المحاماة ومصروفات الدعوى.",
]


class ArabicLegalPleadingTemplate(ArabicTemplateMixin, TestShowcaseTemplate):
    """صحيفة دعوى - Arabic statement of claim"""

    BASE_TEMPLATE = TestShowcaseTemplate
    FONT_CHOICES = ["amiri", "noto_naskh"]
    PAGE_SIZES = ["A4"]

    def get_archetype(self) -> str:
        return "legal"

    def get_category(self) -> str:
        return "legal"

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        ar = self.arabic_setup(seed)
        plaintiff, attorney = ar.person("m"), ar.person()
        defendant = ar.company()["name"]
        city, city2 = ar.city(), ar.city()
        filed = ar.recent_date(200)
        date1 = filed - timedelta(days=ar.rng.randint(200, 500))
        date2 = filed - timedelta(days=ar.rng.randint(20, 90))
        amount1 = round(ar.rng.uniform(50_000, 600_000), -2)
        amount2 = round(ar.rng.uniform(10_000, 150_000), -2)
        contract = ar.digits(f"{date1.year}/{ar.rng.randint(100, 999)}")
        attorney_title = "المحامي" if attorney["gender"] == "m" else "المحامية"
        slots = {
            "P": plaintiff["full"], "D": defendant, "A": f"{attorney_title} {attorney['full']}",
            "city": city, "city2": city2, "cr": ar.digits(ar.rng.randint(10 ** 9, 10 ** 10 - 1)),
            "date1": ar.date_any(date1), "date2": ar.date_any(date2), "contract": contract,
            "service": ar.choice(SERVICES), "period": ar.digits(ar.choice([60, 90, 120, 180])),
            "amount1": ar.money(amount1, word=True), "amount2": ar.money(amount2, word=True),
            "total": ar.money(amount1 + amount2, word=True),
        }

        def numbered(templates, start):
            return [{"num": ar.digits(start + i), "text": t.format(**slots)} for i, t in enumerate(templates)]

        facts = [FACTS[0], FACTS[1]] + ar.sample(FACTS[2:], ar.rng.randint(2, 4))
        sections = [("أولاً: الأطراف", PARTIES), ("ثانياً: الوقائع", facts),
                    ("ثالثاً: الأسانيد النظامية", ar.sample(GROUNDS, ar.rng.randint(2, 3)))]
        numbered_sections, n = [], 1
        for heading, paras in sections:
            numbered_sections.append({"heading": heading, "paragraphs": numbered(paras, n)})
            n += len(paras)

        claims = ar.sample(CLAIMS, 3)
        return {
            "country_name": ar.country["name"],
            "court_name": f"{ar.choice(COURTS)} في {city}",
            "circuit": ar.choice(CIRCUITS),
            "plaintiff_name": plaintiff["full"],
            "defendant_name": defendant,
            "case_number": ar.digits(f"{ar.rng.randint(44, 47)}{ar.rng.randint(1000000, 9999999)}"),
            "document_type": "صحيفة دعوى",
            "claims": [{"num": ar.digits(i), "text": c} for i, c in enumerate(claims, 1)],
            "attorney_title": attorney_title,
            "attorney_name": attorney["full"],
            "law_firm": f"مكتب {attorney['family']} للمحاماة والاستشارات القانونية",
            "license_number": ar.digits(f"{ar.rng.randint(1, 60)}/{ar.rng.randint(100, 999)}"),
            "date_filed": ar.date_any(filed),
            "sections": numbered_sections,
            "requests": [{"num": ar.digits(i), "text": t.format(**slots)} for i, t in enumerate(REQUESTS, 1)],
        }
