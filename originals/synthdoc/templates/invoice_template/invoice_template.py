"""
InvoiceTemplate implementation using the new template system.
"""
from synthdoc.templates.base import BaseTemplate
from typing import Dict, Any
import random


class InvoiceTemplate(BaseTemplate):
    """Template for invoices with line items and tax calculations"""

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        """Generate sample data for invoice"""
        if seed is not None:
            random.seed(seed)
            self.fake.seed_instance(seed)

        service_items = [
            'Consulting Services', 'Professional Services', 'Web Development',
            'Software Development', 'Project Management', 'Design Services',
            'Marketing Services', 'SEO Optimization', 'Content Writing',
            'Technical Support', 'System Maintenance', 'Database Administration',
            'Cloud Hosting Services', 'API Development', 'Mobile App Development',
            'UI/UX Design', 'Quality Assurance Testing', 'Security Audit',
            'Training Services', 'Documentation Services'
        ]

        product_items = [
            'Office Supplies', 'Computer Hardware', 'Software License',
            'Printer Toner', 'Networking Equipment', 'Server Hardware',
            'Backup Storage', 'Security Software', 'Productivity Suite',
            'Collaboration Tools', 'Project Management Software',
            'Antivirus Software', 'Firewall Appliance', 'Router',
            'Switch', 'Cable Management', 'Keyboards', 'Monitors',
            'Webcams', 'Headsets'
        ]

        all_items = service_items + product_items

        tax_rate = self.fake.random_int(min=5, max=10)

        num_items = self.fake.random_int(min=3, max=6)
        items = []
        subtotal = 0

        for i in range(num_items):
            quantity = self.fake.random_int(min=1, max=10)
            unit_price = round(self.fake.random.uniform(25, 500), 2)
            item_discount = round(self.fake.random.uniform(0, 0.1) * unit_price * quantity, 2) if random.random() < 0.2 else 0.0
            amount = round(quantity * unit_price - item_discount, 2)
            subtotal += amount

            items.append({
                'description': self.fake.random_element(all_items),
                'quantity': quantity,
                'unit_price': f"${unit_price:,.2f}",
                'discount': f"${item_discount:,.2f}" if item_discount > 0 else "-",
                'tax_rate': f"{tax_rate}%",
                'amount': f"${amount:,.2f}"
            })

        tax_amount = round(subtotal * tax_rate / 100, 2)

        total_discount = sum(
            float(item['discount'].replace('$', '').replace(',', ''))
            for item in items if item['discount'] != "-"
        )

        total_amount = round(subtotal + tax_amount, 2)

        invoice_date = self.fake.date_this_year()
        due_date = self.fake.date_between(start_date=invoice_date, end_date='+30d')
        signature_date = invoice_date

        payment_terms_options = [
            'Payment due within 30 days of invoice date.',
            'Net 30 terms. Please remit payment within 30 days.',
            'Payment is due upon receipt. Thank you for your prompt payment.',
            'Net 15 terms apply. Payment due within 15 days.',
            'Payment due within 45 days. Late fees may apply after due date.',
            'Net 30. Please make checks payable to company name listed above.',
            'Payment terms: Due upon receipt. All major credit cards accepted.',
            'Net 30 days from invoice date. Please include invoice number with payment.',
            'Payment due within 30 days. Wire transfer information available upon request.',
            'Due on receipt. Thank you for your business.'
        ]

        company_name = self.fake.company()
        company_domain = company_name.lower().replace(' ', '').replace(',', '').replace('.', '')[:15]
        
        bank_name = self.fake.random_element([
            "First National Bank",
            "American Bank & Trust",
            "City Financial Group",
            "Global Trust Bank",
            "Metropolitan Credit Union"
        ])
        account_name = company_name
        account_number = f"****{self.fake.random_int(1000, 9999)}"
        routing_number = f"{self.fake.random_int(100000000, 999999999):09d}"

        status = self.fake.random_element(["Pending", "Paid", "Overdue", "Draft"])
        payment_method = self.fake.random_element(["check", "bank_transfer", "credit_card"])

        terms_pool = [
            "Payment is due within 30 days of the invoice date.",
            "A late fee of 1.5% per month will be applied to overdue balances.",
            "All disputes must be raised within 15 days of receipt.",
            "Returned checks are subject to a $35 processing fee.",
            "Services are non-refundable once delivered and accepted.",
            "Prices are valid for 30 days from the date of this invoice.",
            "Client agrees to pay all collection costs and attorney fees.",
            "This invoice is governed by the laws of the issuing state.",
        ]
        num_terms = self.fake.random_int(min=3, max=5)
        terms = random.sample(terms_pool, min(num_terms, len(terms_pool)))

        return {
            'company_name': company_name,
            'company_address': self.fake.address().replace('\n', ', '),
            'company_contact': self.fake.phone_number(),
            'client_name': self.fake.name(),
            'client_address': self.fake.address().replace('\n', ', '),
            'client_contact': self.fake.phone_number(),
            'invoice_number': f"INV-{self.fake.random_int(min=1000, max=9999)}",
            'invoice_date': invoice_date.strftime('%Y-%m-%d'),
            'due_date': due_date.strftime('%Y-%m-%d'),
            'items': items,
            'subtotal': f"${subtotal:,.2f}",
            'tax_rate': tax_rate,
            'tax_amount': f"${tax_amount:,.2f}",
            'total_amount': f"${total_amount:,.2f}",
            'payment_terms': self.fake.random_element(payment_terms_options),
            'account_name': account_name,
            'account_number': account_number,
            'bank_name': bank_name,
            'routing_number': routing_number,
            'client_email': self.fake.email(),
            'client_phone': self.fake.phone_number(),
            'company_email': f"billing@{company_domain}.com",
            'company_phone': self.fake.phone_number(),
            'company_tagline': self.fake.random_element([
                "Quality Service Since 1995",
                "Your Trusted Partner",
                "Excellence in Every Project",
                "Innovative Solutions",
                ""
            ]),
            'company_website': f"www.{company_domain}.com",
            'signature_date': signature_date.strftime('%Y-%m-%d'),
            'status': status,
            'total_discount': f"${total_discount:,.2f}",
            'terms': terms,
            'payment_method': payment_method,
        }

    def get_template_string(self) -> str:
        """Get template from external files (V2)."""
        html = self._load_file('template.html')
        css = self._load_file('styles.css')
        js = self._load_file('script.js')

        if css and '</head>' in html:
            html = html.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)

        if js and '</body>' in html:
            html = html.replace('</body>', f'<script>\n{js}\n</script>\n</body>', 1)

        return html
