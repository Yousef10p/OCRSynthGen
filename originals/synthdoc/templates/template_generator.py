#!/usr/bin/env python3
"""
Template Generator Script - Creates full template scaffolds for new document types.
Usage: python template_generator.py
"""
import json
import os
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

NEW_TEMPLATES = [
    # Healthcare
    {"name": "radiology_report", "category": "healthcare", "description": "Radiology imaging report (X-ray, CT, MRI)"},
    {"name": "surgical_op_note", "category": "healthcare", "description": "Surgical operative note"},
    {"name": "referral_letter", "category": "healthcare", "description": "Physician referral letter"},
    {"name": "medical_bill", "category": "healthcare", "description": "Medical billing statement"},
    {"name": "eob_detailed", "category": "healthcare", "description": "Detailed Explanation of Benefits"},
    # Real Estate
    {"name": "quitclaim_deed", "category": "real_estate", "description": "Quitclaim deed property transfer"},
    {"name": "title_commitment", "category": "real_estate", "description": "Title insurance commitment"},
    {"name": "property_appraisal", "category": "real_estate", "description": "Real estate appraisal report"},
    {"name": "hoa_agreement", "category": "real_estate", "description": "Homeowners association agreement"},
    {"name": "property_tax_bill", "category": "real_estate", "description": "Property tax assessment bill"},
    {"name": "rental_application", "category": "real_estate", "description": "Residential rental application"},
    {"name": "eviction_notice", "category": "real_estate", "description": "Legal eviction notice"},
    # Financial
    {"name": "credit_application", "category": "financial", "description": "Credit application form"},
    {"name": "loan_disclosure", "category": "financial", "description": "Truth in lending disclosure"},
    {"name": "retirement_statement", "category": "financial", "description": "401k/retirement account statement"},
    {"name": "investment_portfolio", "category": "financial", "description": "Investment portfolio statement"},
    {"name": "wire_transfer", "category": "financial", "description": "Wire transfer instruction form"},
    {"name": "ach_authorization", "category": "financial", "description": "ACH debit authorization form"},
    {"name": "merchant_statement", "category": "financial", "description": "Merchant processing statement"},
    {"name": "tax_1098", "category": "financial", "description": "Mortgage interest 1098 form"},
    # Tax/Government
    {"name": "form_1040es", "category": "financial", "description": "Estimated tax payment form"},
    {"name": "ssa1099", "category": "government", "description": "Social Security benefit statement"},
    {"name": "fbar_form", "category": "government", "description": "FBAR foreign bank account report"},
    {"name": "business_license", "category": "government", "description": "Business license certificate"},
    {"name": "fictitious_business", "category": "government", "description": "Fictitious business name statement"},
    {"name": "passport_renewal", "category": "government", "description": "Passport renewal application"},
    {"name": "ss_benefit_letter", "category": "government", "description": "Social Security benefit letter"},
    {"name": "unemployment_benefit", "category": "government", "description": "Unemployment benefits statement"},
    {"name": "workers_comp", "category": "government", "description": "Workers compensation claim"},
    {"name": "notarized_affidavit", "category": "legal", "description": "Notarized affidavit document"},
    {"name": "llc_formation", "category": "legal", "description": "LLC articles of organization"},
    {"name": "corporate_resolution", "category": "legal", "description": "Corporate resolution document"},
]

PYTHON_TEMPLATE = '''"""
{name_upper} Template - {description}
"""
from synthdoc.templates.base import BaseTemplate
from synthdoc.templates.entity_generator import EntityGenerator
from typing import Dict, Any
import random


class {class_name}(BaseTemplate):
    """Template for {description}"""

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            self.fake.seed_instance(seed)
        
        eg = EntityGenerator(seed)
        
        # Generate entities
        person = eg.generate_person("person", min_age=18, max_age=70)
        org = eg.generate_organization("org", industry="{category}")
        
        # Generate dates
        doc_date = self.fake.date_between(start_date='-30d', end_date='today')
        
        # Generate amounts
        amounts = [eg.generate_amount(min_val=100, max_val=5000) for _ in range(3)]
        
        return {{
            'document_id': eg.generate_id("DOC-", 8),
            'document_date': eg.format_date(doc_date),
            
            'person_name': person.full_name,
            'person_address': person.address.one_line if person.address else "",
            'person_phone': person.phone,
            'person_email': person.email,
            
            'organization_name': org.display_name,
            'organization_address': org.address.one_line if org.address else "",
            'organization_phone': org.phone,
            
            'amount_1': amounts[0],
            'amount_2': amounts[1],
            'amount_3': amounts[2],
            'total_amount': eg.format_amount(
                sum(float(a.replace('$', '').replace(',', '')) for a in amounts)
            ),
            
            'notes': self.fake.sentence(nb_words=15),
            'terms': self.fake.paragraph(nb_sentences=3),
        }}

    def get_template_string(self) -> str:
        html = self._load_file('template.html')
        css = self._load_file('styles.css')
        
        if css and '</head>' in html:
            html = html.replace('</head>', f'<style>\\n{{css}}\\n</style>\\n</head>', 1)
        
        return html
'''

INIT_TEMPLATE = '''"""
{name_upper} Template
"""
from .{name}_template import {class_name}

__all__ = ['{class_name}']
'''

HTML_TEMPLATE = '''<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:schema="http://schema.org/"
      xmlns:doco="http://purl.org/spar/doco/"
      xmlns:doco-fd="http://purl.org/doco-fd/"
      xmlns:foaf="http://xmlns.com/foaf/0.1/"
      xmlns:dc="http://purl.org/dc/terms/">
<head>
    <meta charset="UTF-8">
    <title>{name_upper}</title>
</head>
<body typeof="schema:Document doco:BodyMatter" resource="#{name}">
    <div class="document-container">
        <header typeof="doco-fd:Header" resource="#doc_header" class="document-header">
            <h1 property="doco:title dc:title">{name_upper}</h1>
            <p>Document ID: <span property="schema:identifier">{{ document_id }}</span></p>
            <p>Date: <span property="schema:dateCreated">{{ document_date }}</span></p>
        </header>

        <section typeof="doco-fd:FormSection" resource="#person_section" class="person-section">
            <h2 property="doco:sectionTitle">PERSONAL INFORMATION</h2>
            <div typeof="schema:Person doco-fd:FormField" resource="#person_field">
                <p><strong>Name:</strong> <span property="schema:name foaf:name">{{ person_name }}</span></p>
                <p><strong>Address:</strong> <span property="schema:address">{{ person_address }}</span></p>
                <p><strong>Phone:</strong> <span property="schema:telephone">{{ person_phone }}</span></p>
                <p><strong>Email:</strong> <span property="schema:email">{{ person_email }}</span></p>
            </div>
        </section>

        <section typeof="doco-fd:FormSection" resource="#org_section" class="organization-section">
            <h2 property="doco:sectionTitle">ORGANIZATION</h2>
            <div typeof="schema:Organization doco-fd:FormField" resource="#org_field">
                <p><strong>Name:</strong> <span property="schema:name">{{ organization_name }}</span></p>
                <p><strong>Address:</strong> <span property="schema:address">{{ organization_address }}</span></p>
                <p><strong>Phone:</strong> <span property="schema:telephone">{{ organization_phone }}</span></p>
            </div>
        </section>

        <section typeof="doco-fd:FormSection" resource="#amounts_section" class="amounts-section">
            <h2 property="doco:sectionTitle">AMOUNTS</h2>
            <table typeof="doco:Table" class="amounts-table">
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    <tr typeof="schema:MonetaryAmount">
                        <td>Amount 1</td>
                        <td property="schema:value">{{ amount_1 }}</td>
                    </tr>
                    <tr typeof="schema:MonetaryAmount">
                        <td>Amount 2</td>
                        <td property="schema:value">{{ amount_2 }}</td>
                    </tr>
                    <tr typeof="schema:MonetaryAmount">
                        <td>Amount 3</td>
                        <td property="schema:value">{{ amount_3 }}</td>
                    </tr>
                </tbody>
                <tfoot>
                    <tr typeof="schema:MonetaryAmount" class="total-row">
                        <th>Total</th>
                        <td property="schema:value">{{ total_amount }}</td>
                    </tr>
                </tfoot>
            </table>
        </section>

        <section typeof="doco-fd:FormSection" resource="#notes_section" class="notes-section">
            <h2 property="doco:sectionTitle">NOTES</h2>
            <p property="doco:bodyText">{{ notes }}</p>
        </section>

        <section typeof="doco-fd:FormSection" resource="#terms_section" class="terms-section">
            <h2 property="doco:sectionTitle">TERMS AND CONDITIONS</h2>
            <p property="doco:bodyText">{{ terms }}</p>
        </section>

        <footer typeof="doco-fd:Footer" resource="#doc_footer" class="document-footer">
            <p>This document was generated for testing and demonstration purposes.</p>
        </footer>
    </div>
</body>
</html>
'''

CSS_TEMPLATE = '''.document-container {
    font-family: 'Arial', 'Helvetica', sans-serif;
    font-size: 11pt;
    max-width: 8.5in;
    margin: 0.5in auto;
    padding: 0.5in;
    background: white;
    color: #000;
    line-height: 1.5;
}

.document-header {
    text-align: center;
    border-bottom: 2px solid #000;
    padding-bottom: 15px;
    margin-bottom: 20px;
}

.document-header h1 {
    font-size: 16pt;
    margin: 0 0 10px 0;
    text-transform: uppercase;
}

section {
    margin-bottom: 20px;
    page-break-inside: avoid;
}

section h2 {
    font-size: 12pt;
    font-weight: bold;
    background: #333;
    color: white;
    padding: 5px 10px;
    margin: 0 0 10px 0;
}

.person-section,
.organization-section {
    background: #f9f9f9;
    padding: 10px 15px;
    border: 1px solid #ccc;
}

.amounts-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10pt;
}

.amounts-table th,
.amounts-table td {
    padding: 8px;
    border: 1px solid #ccc;
    text-align: left;
}

.amounts-table th {
    background: #333;
    color: white;
}

.amounts-table .total-row {
    background: #f0f0f0;
    font-weight: bold;
}

.notes-section,
.terms-section {
    padding: 10px;
    background: #f9f9f9;
    border-left: 3px solid #333;
}

.document-footer {
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #000;
    text-align: center;
    font-size: 9pt;
    color: #666;
}
'''


def to_class_name(name: str) -> str:
    parts = name.split('_')
    return ''.join(p.capitalize() for p in parts) + 'Template'


def generate_template(template_info: dict):
    name = template_info['name']
    name_upper = name.replace('_', ' ').upper()
    class_name = to_class_name(name)
    
    template_dir = TEMPLATES_DIR / f"{name}_template"
    template_dir.mkdir(exist_ok=True)
    
    py_file = template_dir / f"{name}_template.py"
    init_file = template_dir / "__init__.py"
    html_file = template_dir / "template.html"
    css_file = template_dir / "styles.css"
    
    py_content = PYTHON_TEMPLATE.format(
        name=name,
        name_upper=name_upper,
        class_name=class_name,
        description=template_info['description'],
        category=template_info['category'],
    )
    
    init_content = INIT_TEMPLATE.format(
        name=name,
        name_upper=name_upper,
        class_name=class_name,
    )
    
    html_content = HTML_TEMPLATE.format(
        name=name,
        name_upper=name_upper,
    )
    
    py_file.write_text(py_content, encoding='utf-8')
    init_file.write_text(init_content, encoding='utf-8')
    html_file.write_text(html_content, encoding='utf-8')
    css_file.write_text(CSS_TEMPLATE, encoding='utf-8')
    (template_dir / "template.json").write_text(
        json.dumps({"name": name, "category": template_info['category'], "class": class_name}, indent=2) + "\n",
        encoding='utf-8',
    )

    print(f"Generated: {name}_template/")
    return class_name


def main():
    generated = []
    for template_info in NEW_TEMPLATES:
        class_name = generate_template(template_info)
        generated.append((template_info['name'], class_name, template_info['category']))
    
    print(f"\nGenerated {len(generated)} templates; each registers itself through its template.json.")


if __name__ == '__main__':
    main()
