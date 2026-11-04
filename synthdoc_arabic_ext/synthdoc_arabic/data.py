"""
Realistic synthetic Arabic data for SynthDoc templates.

Faker's Arabic locales only cover names well: ``ar_SA`` addresses come out as
"70669 فردوس Mission | ناردينbury, OR 13900", companies as "كانو, آل عطفة and ...",
and paragraphs as Latin lorem ipsum. Everything except personal names is therefore
generated here from curated Arabic vocabularies. All organisations, banks and
merchants are fictional.
"""
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

from faker import Faker

try:
    from hijridate import Gregorian
except ImportError:  # Hijri dates are optional
    Gregorian = None


EASTERN_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

GREGORIAN_MONTHS = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]
LEVANTINE_MONTHS = [
    "كانون الثاني", "شباط", "آذار", "نيسان", "أيار", "حزيران",
    "تموز", "آب", "أيلول", "تشرين الأول", "تشرين الثاني", "كانون الأول",
]
HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة",
    "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة",
]
WEEKDAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]

# Country profiles: cities with districts, phone formats, currency and VAT.
COUNTRIES = {
    "SA": {
        "name": "المملكة العربية السعودية",
        "cities": {
            "الرياض": ["العليا", "الملز", "النخيل", "الياسمين", "السليمانية", "الورود", "الربوة", "الملقا"],
            "جدة": ["الروضة", "الحمراء", "الصفا", "النعيم", "الشاطئ", "البوادي", "الزهراء"],
            "مكة المكرمة": ["العزيزية", "الشوقية", "النسيم", "الزاهر"],
            "المدينة المنورة": ["قباء", "العوالي", "الأزهري", "العزيزية"],
            "الدمام": ["الفيصلية", "الشاطئ الغربي", "الجلوية", "المزروعية"],
            "الخبر": ["الثقبة", "العقربية", "الراكة", "اليرموك"],
            "أبها": ["المنسك", "الخالدية", "الموظفين"],
            "بريدة": ["الريان", "الصفراء", "الإسكان"],
        },
        "streets": ["طريق الملك فهد", "طريق الملك عبدالعزيز", "شارع الأمير سلطان", "شارع التحلية",
                    "شارع العروبة", "شارع الستين", "طريق الأمير ماجد", "شارع الأمير محمد بن عبدالعزيز",
                    "شارع أبي بكر الصديق", "شارع عمر بن الخطاب", "شارع ابن خلدون", "شارع الفارابي"],
        "address_style": "national",
        "mobile": "+966 5{a} {b} {c}", "landline": "01{a} {b} {c}",
        "currency": "ر.س", "currency_word": "ريال سعودي", "vat": 15, "iban": "SA", "iban_len": 22,
        "id_label": "رقم الهوية الوطنية",
    },
    "AE": {
        "name": "الإمارات العربية المتحدة",
        "cities": {
            "دبي": ["ديرة", "بر دبي", "القصيص", "الكرامة", "جميرا", "البرشاء"],
            "أبوظبي": ["الخالدية", "المشرف", "البطين", "الزاهية"],
            "الشارقة": ["النهدة", "المجاز", "القاسمية"],
        },
        "streets": ["شارع الشيخ زايد", "شارع الوصل", "شارع المطار", "شارع الاتحاد", "شارع الكورنيش",
                    "شارع حمدان بن محمد", "شارع الخليج العربي"],
        "address_style": "pobox",
        "mobile": "+971 5{a} {b} {c}", "landline": "04 {b} {c}",
        "currency": "د.إ", "currency_word": "درهم إماراتي", "vat": 5, "iban": "AE", "iban_len": 21,
        "id_label": "رقم الهوية الإماراتية",
    },
    "EG": {
        "name": "جمهورية مصر العربية",
        "cities": {
            "القاهرة": ["مدينة نصر", "المعادي", "الزمالك", "مصر الجديدة", "التجمع الخامس"],
            "الإسكندرية": ["سموحة", "سيدي جابر", "رشدي", "العجمي"],
            "الجيزة": ["المهندسين", "الدقي", "الهرم", "الشيخ زايد"],
        },
        "streets": ["شارع التحرير", "شارع جامعة الدول العربية", "شارع عباس العقاد", "شارع مكرم عبيد",
                    "شارع الثورة", "طريق الكورنيش", "شارع طلعت حرب"],
        "address_style": "egypt",
        "mobile": "+20 1{a} {b} {c}", "landline": "02 {b} {c}",
        "currency": "ج.م", "currency_word": "جنيه مصري", "vat": 14, "iban": "EG", "iban_len": 27,
        "id_label": "الرقم القومي",
    },
    "JO": {
        "name": "المملكة الأردنية الهاشمية",
        "cities": {
            "عمّان": ["عبدون", "الشميساني", "جبل عمّان", "خلدا", "الصويفية", "تلاع العلي"],
            "إربد": ["الحصن", "حي التركمان", "البارحة"],
            "الزرقاء": ["الزرقاء الجديدة", "حي معصوم"],
        },
        "streets": ["شارع الملكة رانيا", "شارع مكة", "شارع وصفي التل", "شارع الجاردنز",
                    "شارع الأمير حسن", "شارع زهران"],
        "address_style": "egypt",
        "mobile": "+962 7{a} {b} {c}", "landline": "06 {b} {c}",
        "currency": "د.أ", "currency_word": "دينار أردني", "vat": 16, "iban": "JO", "iban_len": 28,
        "id_label": "الرقم الوطني",
    },
}

# Fictional company name parts (core word -> Latin slug used for e-mail/web domains)
COMPANY_CORES = {
    "الأفق": "alofuq", "النخبة": "alnukhba", "الريادة": "alriyada", "الواحة": "alwaha",
    "المستقبل": "almustaqbal", "البيان": "albayan", "الصفوة": "alsafwa", "الإتقان": "aletqan",
    "المدار": "almadar", "رؤية": "ruya", "السنابل": "alsanabel", "الرواد": "alruwad",
    "الماسة": "almasa", "الشروق": "alshorouq", "الينابيع": "alyanabee", "القمة": "alqimma",
}
COMPANY_PREFIXES = ["شركة", "مؤسسة", "مجموعة", "مكتب"]
COMPANY_ACTIVITIES = [
    "للتجارة", "للمقاولات العامة", "للحلول التقنية", "للاستشارات الإدارية",
    "للخدمات اللوجستية", "للتسويق الرقمي", "للأنظمة المتقدمة", "للتطوير العقاري",
    "للصناعات الغذائية", "للتدريب والتطوير", "للتجهيزات الطبية",
]
COMPANY_SUFFIXES = ["", "", "المحدودة", "ذ.م.م", "القابضة"]
COMPANY_TAGLINES = [
    "جودة وخبرة منذ عام ١٩٩٥", "شريككم الموثوق", "التميز في كل مشروع",
    "حلول مبتكرة لأعمالكم", "نبني المستقبل معاً", "",
]

BANKS = {  # fictional banks
    "بنك الواحة الوطني": "alwahabank", "مصرف النخيل الإسلامي": "alnakheelbank",
    "بنك المدار التجاري": "almadarbank", "مصرف السنابل": "alsanabelbank",
    "بنك الريادة الخليجي": "alriyadabank", "مصرف الأمانة المتحد": "alamanabank",
    "بنك الرؤية الأول": "ruyabank", "البنك العربي للتنمية والاستثمار": "adibank",
}

INVOICE_SERVICES = [
    "خدمات استشارية", "تطوير موقع إلكتروني", "تطوير تطبيق جوال", "إدارة مشاريع",
    "تصميم الهوية البصرية", "خدمات التسويق الرقمي", "تحسين محركات البحث", "كتابة المحتوى",
    "الدعم الفني السنوي", "صيانة الأنظمة", "إدارة قواعد البيانات", "استضافة سحابية",
    "تطوير واجهات برمجية", "تصميم تجربة المستخدم", "اختبار جودة البرمجيات", "تدقيق أمن المعلومات",
    "دورة تدريبية للموظفين", "إعداد الوثائق الفنية",
]
INVOICE_PRODUCTS = [
    "أدوات مكتبية", "أجهزة حاسب محمول", "رخصة برنامج محاسبة", "حبر طابعة", "معدات شبكات",
    "خادم رئيسي", "وحدة تخزين احتياطي", "برنامج مكافحة الفيروسات", "جدار حماية", "موجه شبكة",
    "شاشة عرض ٢٧ بوصة", "لوحة مفاتيح عربية", "كاميرا مراقبة", "سماعات رأس", "كراسي مكتبية",
]
INVOICE_PAYMENT_TERMS = [
    "يستحق السداد خلال ٣٠ يوماً من تاريخ الفاتورة.",
    "يُرجى السداد خلال ١٥ يوماً من تاريخ الاستلام.",
    "الدفع مستحق عند الاستلام، شكراً لسرعة السداد.",
    "يستحق السداد خلال ٤٥ يوماً، وقد تُطبق غرامة تأخير بعد تاريخ الاستحقاق.",
    "يُرجى ذكر رقم الفاتورة عند إجراء التحويل البنكي.",
]
INVOICE_TERMS = [
    "يستحق سداد قيمة الفاتورة خلال ثلاثين يوماً من تاريخ إصدارها.",
    "تُفرض غرامة تأخير بنسبة ١٫٥٪ شهرياً على المبالغ المتأخرة.",
    "يجب تقديم أي اعتراض على الفاتورة خلال خمسة عشر يوماً من استلامها.",
    "الخدمات المقدمة غير قابلة للاسترداد بعد تسليمها وقبولها.",
    "الأسعار الواردة في هذه الفاتورة سارية لمدة ثلاثين يوماً.",
    "يتحمل العميل جميع تكاليف التحصيل والأتعاب القانونية عند التأخر في السداد.",
    "تخضع هذه الفاتورة للأنظمة والقوانين المعمول بها في بلد الإصدار.",
    "جميع الأسعار شاملة ضريبة القيمة المضافة ما لم يُذكر خلاف ذلك.",
]

FIRST_NAMES_MALE_EXTRA = ["محمد", "أحمد", "عبدالله", "خالد", "فهد", "سعود", "عمر", "يوسف", "إبراهيم",
                          "سلطان", "ناصر", "فيصل", "طارق", "ماجد", "حسن", "علي", "بندر", "تركي"]
FIRST_NAMES_FEMALE_EXTRA = ["نورة", "سارة", "ريم", "هيفاء", "لمى", "فاطمة", "مريم", "عبير",
                            "جواهر", "دانة", "أمل", "هند", "منيرة", "رزان"]

DOCTOR_SPECIALTIES = ["طب الأسرة", "الباطنية", "طب الأطفال", "القلب", "العظام", "الجلدية", "الغدد الصماء"]
MEDICATIONS = [
    "ميتفورمين ٥٠٠ ملغ مرتين يومياً", "ليسينوبريل ١٠ ملغ مرة يومياً", "أتورفاستاتين ٢٠ ملغ مساءً",
    "أملوديبين ٥ ملغ صباحاً", "أوميبرازول ٢٠ ملغ قبل الإفطار", "ليفوثيروكسين ٥٠ ميكروغرام صباحاً",
    "فيتامين د ٥٠ ألف وحدة أسبوعياً", "باراسيتامول ٥٠٠ ملغ عند اللزوم", "أسبرين ٨١ ملغ يومياً",
]
VISIT_REASONS = [
    "فحص دوري شامل وتحاليل مخبرية روتينية",
    "متابعة مستوى السكر التراكمي وضبط جرعات العلاج",
    "آلام متكررة في أسفل الظهر منذ أسبوعين",
    "ارتفاع في درجة الحرارة مع سعال جاف منذ ثلاثة أيام",
    "صداع مزمن واضطراب في النوم",
    "متابعة ضغط الدم بعد تعديل العلاج",
]
ALLERGIES = ["لا توجد حساسية معروفة", "حساسية من البنسلين", "حساسية من الفول السوداني",
             "حساسية موسمية من الغبار وحبوب اللقاح", "حساسية من مضادات الالتهاب غير الستيرويدية"]
CHRONIC_CONDITIONS = ["السكري", "ارتفاع ضغط الدم", "الربو", "أمراض القلب", "قصور الغدة الدرقية",
                      "ارتفاع الكوليسترول"]
RELATIONSHIPS = ["الزوج", "الزوجة", "الأب", "الأم", "الأخ", "الأخت", "الابن", "الابنة"]
INSURERS = ["شركة الرعاية للتأمين التعاوني", "الشركة الوطنية للتأمين الصحي", "شركة الأمان للتأمين",
            "شركة الضمان المتحدة للتأمين"]

JOB_TITLES = ["مهندس برمجيات أول", "محاسب قانوني", "مدير مشاريع", "أخصائي موارد بشرية",
              "مصمم جرافيك", "محلل بيانات", "مدير مبيعات", "مهندس شبكات", "مستشار قانوني",
              "أخصائي تسويق رقمي", "مشرف عمليات"]
NATIONALITIES = ["سعودي", "مصري", "أردني", "سوداني", "يمني", "سوري", "مغربي", "تونسي", "لبناني"]
NATIONALITIES_F = {"سعودي": "سعودية", "مصري": "مصرية", "أردني": "أردنية", "سوداني": "سودانية",
                   "يمني": "يمنية", "سوري": "سورية", "مغربي": "مغربية", "تونسي": "تونسية",
                   "لبناني": "لبنانية"}
DEPARTMENTS = ["إدارة تقنية المعلومات", "الإدارة المالية", "إدارة الموارد البشرية",
               "إدارة المبيعات والتسويق", "إدارة العمليات", "الإدارة القانونية"]


class ArabicData:
    """Seeded generator of Arabic document data for one document.

    A document picks a country profile and a digit style (Western 0-9 or
    Arabic-Indic ٠-٩) once, and every number it formats follows that choice.
    """

    def __init__(self, seed: Optional[int] = None, country: Optional[str] = None,
                 eastern_digits: Optional[bool] = None):
        self.rng = random.Random(seed)
        self.fake = Faker("ar_SA")
        if seed is not None:
            self.fake.seed_instance(seed)
        self.country_code = country or self.rng.choices(list(COUNTRIES), weights=[6, 2, 2, 1])[0]
        self.country = COUNTRIES[self.country_code]
        self.eastern = self.rng.random() < 0.4 if eastern_digits is None else eastern_digits

    # -- numbers -----------------------------------------------------------
    def digits(self, value) -> str:
        """Render digits in this document's digit style."""
        s = str(value)
        return s.translate(EASTERN_DIGITS) if self.eastern else s

    def number(self, value: float, decimals: int = 2) -> str:
        s = f"{value:,.{decimals}f}"
        if self.eastern:
            s = s.replace(",", "٬").replace(".", "٫").translate(EASTERN_DIGITS)
        return s

    def money(self, value: float, word: bool = False) -> str:
        unit = self.country["currency_word"] if word else self.country["currency"]
        return f"{self.number(value)} {unit}"

    def percent(self, value) -> str:
        return f"{self.digits(value)}٪" if self.eastern else f"{value}%"

    def ref(self, prefix: str, digits: int = 6) -> str:
        """An alphanumeric reference such as INV-2026-004512 (always Western digits)."""
        return f"{prefix}{self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"

    # -- dates -------------------------------------------------------------
    def date_between(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, max(0, (end - start).days)))

    def recent_date(self, days_back: int = 180) -> date:
        today = date(2026, 10, 31)
        return today - timedelta(days=self.rng.randint(0, days_back))

    def date_numeric(self, d: date) -> str:
        return self.digits(d.strftime("%Y/%m/%d"))

    def date_long(self, d: date, levantine: bool = False) -> str:
        months = LEVANTINE_MONTHS if levantine else GREGORIAN_MONTHS
        return f"{self.digits(d.day)} {months[d.month - 1]} {self.digits(d.year)}م"

    def date_hijri(self, d: date) -> str:
        if Gregorian is None:
            return self.date_long(d)
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        return f"{self.digits(h.day)} {HIJRI_MONTHS[h.month - 1]} {self.digits(h.year)}هـ"

    def date_any(self, d: date) -> str:
        """A date in one of the styles common on Arabic documents."""
        style = self.rng.choice(["numeric", "long", "long"] + (["hijri"] if self.country_code == "SA" else []))
        if style == "numeric":
            return self.date_numeric(d)
        if style == "hijri":
            return self.date_hijri(d)
        return self.date_long(d, levantine=self.country_code == "JO")

    def weekday(self, d: date) -> str:
        return WEEKDAYS[d.weekday()]

    # -- people ------------------------------------------------------------
    def first_name(self, gender: Optional[str] = None) -> str:
        gender = gender or self.rng.choice("mf")
        if gender == "m":
            return self.rng.choice([self.fake.first_name_male(), self.rng.choice(FIRST_NAMES_MALE_EXTRA)])
        return self.rng.choice([self.fake.first_name_female(), self.rng.choice(FIRST_NAMES_FEMALE_EXTRA)])

    def family_name(self) -> str:
        return self.fake.last_name()

    def full_name(self, gender: Optional[str] = None, parts: int = 3) -> str:
        """Arabic full name: given name, father's name(s), family name."""
        names = [self.first_name(gender)]
        names += [self.first_name("m") for _ in range(parts - 2)]
        names.append(self.family_name())
        return " ".join(names)

    def person(self, gender: Optional[str] = None) -> Dict[str, str]:
        gender = gender or self.rng.choice("mf")
        first, father, family = self.first_name(gender), self.first_name("m"), self.family_name()
        return {"first": first, "father": father, "family": family, "gender": gender,
                "full": f"{first} {father} {family}"}

    def national_id(self) -> str:
        return self.digits(f"{self.rng.choice('12')}{self.rng.randint(10 ** 8, 10 ** 9 - 1)}")

    # -- places ------------------------------------------------------------
    def city(self) -> str:
        return self.rng.choice(list(self.country["cities"]))

    def address(self, city: Optional[str] = None) -> str:
        city = city or self.city()
        district = self.rng.choice(self.country["cities"][city])
        street = self.rng.choice(self.country["streets"])
        d = self.digits
        style = self.country["address_style"]
        if style == "national":  # Saudi National Address
            return (f"{d(self.rng.randint(1000, 9999))} {street}، حي {district}، "
                    f"{city} {d(self.rng.randint(11000, 34999))}")
        if style == "pobox":
            return f"مبنى {d(self.rng.randint(1, 250))}، {street}، {district}، {city}، ص.ب {d(self.rng.randint(1000, 99999))}"
        return f"{d(self.rng.randint(1, 120))} {street}، {district}، {city}"

    def phone(self, mobile: bool = True) -> str:
        fmt = self.country["mobile" if mobile else "landline"]
        return fmt.format(a=self.rng.randint(0, 9), b=self.rng.randint(100, 999), c=self.rng.randint(1000, 9999))

    # -- organisations -----------------------------------------------------
    def company(self) -> Dict[str, str]:
        core = self.rng.choice(list(COMPANY_CORES))
        name = f"{self.rng.choice(COMPANY_PREFIXES)} {core} {self.rng.choice(COMPANY_ACTIVITIES)}"
        suffix = self.rng.choice(COMPANY_SUFFIXES)
        if suffix:
            name += f" {suffix}"
        slug = COMPANY_CORES[core]
        return {"name": name, "domain": f"{slug}.com.sa" if self.country_code == "SA" else f"{slug}.com"}

    def bank(self) -> Dict[str, str]:
        name = self.rng.choice(list(BANKS))
        return {"name": name, "domain": f"{BANKS[name]}.com"}

    def iban(self, masked: bool = False) -> str:
        n = self.country["iban_len"] - 4
        body = "".join(str(self.rng.randint(0, 9)) for _ in range(n))
        iban = f"{self.country['iban']}{self.rng.randint(10, 99)}{body}"
        if masked:
            iban = iban[:4] + "*" * (len(iban) - 8) + iban[-4:]
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))

    def email(self, person: Optional[Dict[str, str]] = None, domain: Optional[str] = None) -> str:
        user = self.rng.choice(["info", "contact", "billing", "accounts", "sales", "admin"])
        if person is not None:
            user = f"{self.rng.choice(['user', 'ali', 'm', 'k', 'n', 's', 'h'])}{self.rng.randint(10, 999)}"
        domain = domain or self.rng.choice(["example.com", "mail.example.net", "example.org"])
        return f"{user}@{domain}"

    def choice(self, seq):
        return self.rng.choice(seq)

    def sample(self, seq, k: int) -> List:
        return self.rng.sample(seq, min(k, len(seq)))
