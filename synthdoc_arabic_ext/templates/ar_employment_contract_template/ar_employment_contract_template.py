"""
Arabic employment contract (عقد عمل), from the employment-contract section of
SynthDoc's ``test_showcase`` template: a preamble paragraph, numbered clauses
with justified paragraphs, and a two-party signature block.
"""
from datetime import timedelta
from typing import Any, Dict

from synthdoc.templates.test_showcase_template.test_showcase_template import TestShowcaseTemplate
from synthdoc_arabic.base import ArabicTemplateMixin
from synthdoc_arabic.data import DEPARTMENTS, JOB_TITLES, NATIONALITIES, NATIONALITIES_F

ORDINALS = ["الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع", "الثامن", "التاسع"]
MANAGER_TITLES = ["الرئيس التنفيذي", "المدير العام", "مدير الموارد البشرية"]


class ArabicEmploymentContractTemplate(ArabicTemplateMixin, TestShowcaseTemplate):
    """عقد عمل - Arabic employment contract"""

    BASE_TEMPLATE = TestShowcaseTemplate
    FONT_CHOICES = ["amiri", "noto_naskh", "ibm_plex"]
    PAGE_SIZES = ["A4"]

    def get_archetype(self) -> str:
        return "legal"

    def get_category(self) -> str:
        return "business"

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        ar = self.arabic_setup(seed)
        employer = ar.company()
        manager, employee = ar.person(), ar.person()
        signed = ar.recent_date(120)
        start = signed + timedelta(days=ar.rng.randint(7, 45))
        salary = ar.rng.randrange(6000, 32000, 250)
        housing = round(salary * 0.25)
        transport = round(salary * 0.10)
        nationality = ar.choice(NATIONALITIES)
        is_male = employee["gender"] == "m"
        years = ar.choice([1, 2, 3])
        g = {  # gendered word forms for the employee
            "party": "الطرف الثاني",
            "pronoun": "يلتزم" if is_male else "تلتزم",
            "entitled": "يستحق" if is_male else "تستحق",
            "agrees": "يوافق" if is_male else "توافق",
        }
        job = ar.choice(JOB_TITLES)
        dept = ar.choice(DEPARTMENTS)
        clauses = [
            ("المسمى الوظيفي والمهام",
             f"{g['pronoun']} الطرف الثاني بالعمل لدى الطرف الأول بوظيفة {job} في {dept}، وأداء المهام "
             f"والمسؤوليات المرتبطة بهذه الوظيفة وفق الوصف الوظيفي المعتمد، وما يُسند إليه من أعمال تتناسب "
             f"مع مؤهلاته وخبراته."),
            ("مدة العقد",
             f"مدة هذا العقد {ar.digits(years)} {'سنة' if years == 1 else 'سنوات' if years > 2 else 'سنتان'} "
             f"تبدأ من تاريخ مباشرة العمل في {ar.date_any(start)}، وتتجدد تلقائياً لمدة مماثلة ما لم يُخطر "
             f"أحد الطرفين الآخر كتابياً برغبته في عدم التجديد قبل ستين يوماً من تاريخ انتهائه. ويخضع "
             f"الطرف الثاني لفترة تجربة مدتها {ar.digits(90)} يوماً."),
            ("الأجر والبدلات",
             f"{g['entitled']} الطرف الثاني راتباً أساسياً شهرياً قدره {ar.money(salary, word=True)}، "
             f"وبدل سكن قدره {ar.money(housing, word=True)}، وبدل نقل قدره {ar.money(transport, word=True)}، "
             f"تُصرف في نهاية كل شهر ميلادي بالتحويل إلى حسابه البنكي."),
            ("ساعات العمل والإجازات",
             f"تكون ساعات العمل الفعلية {ar.digits(ar.choice([40, 45, 48]))} ساعة أسبوعياً موزعة على خمسة أيام، "
             f"و{g['entitled']} الطرف الثاني إجازة سنوية مدفوعة الأجر مدتها {ar.digits(ar.choice([21, 30]))} يوماً، "
             f"إضافة إلى الإجازات الرسمية المقررة نظاماً."),
            ("المزايا",
             f"{g['entitled']} الطرف الثاني تأميناً طبياً له ولأفراد أسرته، وتذكرة سفر سنوية، ومكافأة نهاية "
             f"الخدمة وفقاً لأحكام نظام العمل."),
            ("السرية وعدم المنافسة",
             f"{g['pronoun']} الطرف الثاني بالمحافظة على سرية المعلومات والبيانات التي يطلع عليها بحكم عمله، "
             f"وعدم إفشائها لأي جهة خلال مدة العقد وبعد انتهائه، وعدم العمل لدى منشأة منافسة في "
             f"{ar.country['name']} لمدة سنة من تاريخ انتهاء العقد."),
            ("إنهاء العقد",
             "يجوز لأي من الطرفين إنهاء هذا العقد بعد انتهاء فترة التجربة بموجب إشعار كتابي مدته ثلاثون يوماً، "
             "مع مراعاة الحقوق المستحقة لكل طرف وفقاً للأنظمة المعمول بها."),
            ("النظام الواجب التطبيق",
             f"يخضع هذا العقد لأحكام نظام العمل المعمول به في {ar.country['name']}، وتختص الجهات القضائية "
             f"المختصة بالفصل في أي نزاع ينشأ عن تفسيره أو تنفيذه."),
        ]
        keep = [0, 1, 2] + sorted(ar.sample(range(3, len(clauses)), ar.rng.randint(3, 5)))
        clauses = [clauses[i] for i in keep]

        return {
            "contract_number": f"HR-{signed.year}-{ar.rng.randint(100, 999)}",
            "sign_date": ar.date_any(signed),
            "sign_weekday": ar.weekday(signed),
            "city": ar.city(),
            "employer_name": employer["name"],
            "employer_cr": ar.digits(ar.rng.randint(10 ** 9, 10 ** 10 - 1)),
            "employer_address": ar.address(),
            "manager_name": manager["full"],
            "manager_title": ar.choice(MANAGER_TITLES),
            "employee_name": employee["full"],
            "employee_nationality": nationality if is_male else NATIONALITIES_F[nationality],
            "id_label": ar.country["id_label"],
            "employee_id": ar.national_id(),
            "employee_phone": ar.phone(),
            "employee_address": ar.address(),
            "agrees": g["agrees"],
            "clauses": [{"ordinal": f"البند {ORDINALS[i]}", "title": t, "text": body}
                        for i, (t, body) in enumerate(clauses)],
            "copies": ar.digits(2),
        }
