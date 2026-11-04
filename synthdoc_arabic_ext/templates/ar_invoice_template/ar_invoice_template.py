"""
Arabic tax invoice: SynthDoc's ``invoice`` template with RTL layout and Arabic data.
"""
from datetime import timedelta
from typing import Any, Dict

from synthdoc.templates.invoice_template.invoice_template import InvoiceTemplate
from synthdoc_arabic.base import ArabicTemplateMixin
from synthdoc_arabic.data import (COMPANY_TAGLINES, INVOICE_PAYMENT_TERMS, INVOICE_PRODUCTS,
                                  INVOICE_SERVICES, INVOICE_TERMS)

STATUSES = {"paid": "مدفوعة", "overdue": "متأخرة", "draft": "مسودة", "sent": "مُرسلة"}


class ArabicInvoiceTemplate(ArabicTemplateMixin, InvoiceTemplate):
    """فاتورة ضريبية - Arabic invoice with line items and VAT"""

    BASE_TEMPLATE = InvoiceTemplate
    FONT_CHOICES = ["cairo", "tajawal", "ibm_plex", "noto_kufi"]
    PAGE_SIZES = ["A4", "letter"]

    def get_archetype(self) -> str:
        return "invoice"

    def get_category(self) -> str:
        return "financial"

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        ar = self.arabic_setup(seed)
        vat = ar.country["vat"]

        items, subtotal, total_discount = [], 0.0, 0.0
        for description in ar.sample(INVOICE_SERVICES + INVOICE_PRODUCTS, ar.rng.randint(3, 6)):
            quantity = ar.rng.randint(1, 10)
            unit_price = round(ar.rng.uniform(25, 2500), 2)
            discount = round(ar.rng.uniform(0, 0.1) * unit_price * quantity, 2) if ar.rng.random() < 0.25 else 0.0
            amount = round(quantity * unit_price - discount, 2)
            subtotal += amount
            total_discount += discount
            items.append({
                "index": ar.digits(len(items) + 1),
                "description": description,
                "quantity": ar.digits(quantity),
                "unit_price": ar.number(unit_price),
                "discount": ar.number(discount) if discount else "-",
                "tax_rate": ar.percent(vat),
                "amount": ar.number(amount),
            })
        tax_amount = round(subtotal * vat / 100, 2)

        invoice_date = ar.recent_date(300)
        due_date = invoice_date + timedelta(days=ar.choice([15, 30, 45]))
        company, client = ar.company(), ar.person()
        bank = ar.bank()
        status = ar.choice(list(STATUSES))

        return {
            "company_name": company["name"],
            "company_tagline": ar.choice(COMPANY_TAGLINES),
            "company_address": ar.address(),
            "company_email": ar.email(domain=company["domain"]),
            "company_phone": ar.phone(mobile=False),
            "company_website": f"www.{company['domain']}",
            "vat_number": ar.digits(f"3{ar.rng.randint(10 ** 12, 10 ** 13 - 1)}03"),
            "client_name": client["full"],
            "client_address": ar.address(),
            "client_email": ar.email(person=client),
            "client_phone": ar.phone(),
            "invoice_number": f"INV-{invoice_date.year}-{ar.rng.randint(1000, 9999)}",
            "invoice_date": ar.date_any(invoice_date),
            "due_date": ar.date_any(due_date),
            "items": items,
            "currency": ar.country["currency"],
            "subtotal": ar.money(subtotal),
            "total_discount": ar.money(total_discount),
            "tax_rate": ar.percent(vat),
            "tax_amount": ar.money(tax_amount),
            "total_amount": ar.money(subtotal + tax_amount),
            "payment_terms": ar.choice(INVOICE_PAYMENT_TERMS),
            "account_name": company["name"],
            "bank_name": bank["name"],
            "account_number": ar.iban(masked=True),
            "swift_code": f"{bank['domain'][:4].upper()}{ar.country_code}{ar.choice(['RI', 'JE', 'DU', 'CA'])}",
            "status": status,
            "status_label": STATUSES[status],
            "payment_method": ar.choice(["check", "bank_transfer", "credit_card"]),
            "terms": [{"num": ar.digits(i), "text": t}
                      for i, t in enumerate(ar.sample(INVOICE_TERMS, ar.rng.randint(3, 5)), 1)],
            "signature_date": ar.date_numeric(invoice_date),
        }
