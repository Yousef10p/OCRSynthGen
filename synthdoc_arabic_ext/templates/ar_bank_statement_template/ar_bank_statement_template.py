"""
Arabic bank statement: SynthDoc's ``bank_statement`` template with RTL layout and Arabic data.
"""
from datetime import timedelta
from typing import Any, Dict

from synthdoc.templates.bank_statement_template.bank_statement_template import BankStatementTemplate
from synthdoc_arabic.base import ArabicTemplateMixin

# (description, weight, category slug)
DEPOSITS = [("إيداع راتب", 20, "income"), ("تحويل وارد", 20, "transfer"), ("إيداع نقدي", 10, "income"),
            ("أرباح حساب الادخار", 5, "income"), ("استرداد مبلغ مشتريات", 10, "transfer"),
            ("حوالة واردة من الخارج", 8, "transfer")]
WITHDRAWALS = [("سحب نقدي من الصراف الآلي", 15, "fees"), ("شراء عبر الإنترنت", 15, "shopping"),
               ("دفع فاتورة", 15, "utilities"), ("رسوم خدمات بنكية", 3, "fees"),
               ("قسط تمويل شخصي", 10, "other"), ("شراء بالبطاقة", 22, "shopping"),
               ("تحويل صادر", 10, "transfer")]
MERCHANTS = {  # fictional merchants -> category slug
    "أسواق الخير المركزية": "shopping", "صيدلية الشفاء": "healthcare", "مطاعم السنبلة": "food",
    "مقهى الركن الهادئ": "food", "محطة وقود الطريق": "transport", "مكتبة المعرفة": "shopping",
    "متجر الإلكترونيات الحديثة": "shopping", "تطبيق توصيل سريع": "food",
    "منصة البث المرئي": "entertainment", "تطبيق النقل الذكي": "transport",
}
PAYEES = {"شركة الكهرباء": "utilities", "مصلحة المياه": "utilities", "شركة الاتصالات": "utilities",
          "التأمين الطبي": "healthcare", "الرسوم المدرسية": "other"}
CATEGORY_NAMES = {"shopping": "تسوق", "food": "مطاعم ومقاهٍ", "transport": "مواصلات",
                  "utilities": "فواتير وخدمات", "healthcare": "صحة", "entertainment": "ترفيه",
                  "income": "دخل", "fees": "رسوم", "transfer": "تحويلات", "other": "أخرى"}
SERVICE_HOURS = ["الأحد - الخميس، ٨:٠٠ ص - ٤:٠٠ م", "على مدار الساعة طوال أيام الأسبوع",
                 "السبت - الخميس، ٩:٠٠ ص - ٩:٠٠ م"]


class ArabicBankStatementTemplate(ArabicTemplateMixin, BankStatementTemplate):
    """كشف حساب بنكي - Arabic bank statement"""

    BASE_TEMPLATE = BankStatementTemplate
    FONT_CHOICES = ["ibm_plex", "tajawal", "cairo", "noto_kufi"]
    PAGE_SIZES = ["A4", "letter"]

    def get_archetype(self) -> str:
        return "ledger"

    def get_category(self) -> str:
        return "financial"

    def _pick(self, ar, weighted):
        return ar.rng.choices(weighted, weights=[w for _, w, _ in weighted], k=1)[0]

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        ar = self.arabic_setup(seed)
        end_date = ar.recent_date(150)
        start_date = end_date - timedelta(days=ar.rng.randint(28, 31))
        opening = round(ar.rng.uniform(3000, 40000), 2)
        balance = opening

        transactions, deposits, withdrawals, by_category = [], 0.0, 0.0, {}
        dates = sorted(ar.date_between(start_date, end_date) for _ in range(ar.rng.randint(8, 12)))
        for d in dates:
            is_deposit = balance < 1500 or (balance < 60000 and ar.rng.random() < 0.35)
            if is_deposit:
                desc, _, cat = self._pick(ar, DEPOSITS)
                amount = round(ar.rng.uniform(4000, 18000) if "راتب" in desc else ar.rng.uniform(150, 6000), 2)
                balance += amount
                deposits += amount
                is_merchant = False
            else:
                desc, _, cat = self._pick(ar, WITHDRAWALS)
                is_merchant = False
                if desc in ("شراء عبر الإنترنت", "شراء بالبطاقة"):
                    merchant = ar.choice(list(MERCHANTS))
                    desc, cat, is_merchant = f"{desc} - {merchant}", MERCHANTS[merchant], True
                elif desc == "دفع فاتورة":
                    payee = ar.choice(list(PAYEES))
                    desc, cat, is_merchant = f"{desc} - {payee}", PAYEES[payee], True
                amount = round(min(ar.rng.uniform(15, 3500), balance - 100), 2)
                if amount <= 5:
                    continue
                balance -= amount
                withdrawals += amount
                by_category[cat] = by_category.get(cat, 0.0) + amount
            transactions.append({
                "date": ar.date_numeric(d),
                "description": desc,
                "is_merchant": is_merchant,
                "ref_code": f"FT{ar.rng.randint(10 ** 9, 10 ** 10 - 1)}",
                "category": CATEGORY_NAMES[cat],
                "category_slug": cat,
                "debit_amt": "" if is_deposit else ar.number(amount),
                "credit_amt": ar.number(amount) if is_deposit else "",
                "running_bal": ar.number(balance),
            })

        category_summary = [
            {"name": CATEGORY_NAMES[c], "slug": c, "amount": ar.number(a),
             "percentage": round(a / withdrawals * 100), "percentage_label": ar.percent(round(a / withdrawals * 100))}
            for c, a in sorted(by_category.items(), key=lambda kv: -kv[1])
        ] if withdrawals else []

        bank = ar.bank()
        return {
            "bank_name": bank["name"],
            "bank_address": ar.address(),
            "account_holder": ar.full_name(),
            "account_number": ar.iban(masked=True),
            "account_type": ar.choice(["حساب جاري", "حساب توفير", "حساب رواتب"]),
            "currency": ar.country["currency_word"],
            "start_date": ar.date_numeric(start_date),
            "end_date": ar.date_numeric(end_date),
            "beginning_balance": ar.money(opening),
            "ending_balance": ar.money(balance),
            "total_deposits": ar.money(deposits),
            "total_withdrawals": ar.money(withdrawals),
            "transactions": transactions,
            "category_summary": category_summary,
            "customer_service_phone": ar.choice(["8001", "9200"]) + str(ar.rng.randint(10000, 99999)),
            "customer_service_email": f"care@{bank['domain']}",
            "bank_website": f"www.{bank['domain']}",
            "service_hours": ar.choice(SERVICE_HOURS),
            "routing_number": f"{bank['domain'][:4].upper()}{ar.country_code}{ar.choice(['RI', 'JE', 'DU'])}",
            "statement_id": f"STMT-{ar.rng.randint(100000, 999999)}",
            "generated_date": ar.date_any(end_date + timedelta(days=1)),
        }
