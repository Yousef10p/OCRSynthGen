"""
Test Showcase Template - A visual demonstration template for testing NER metrics and bounding boxes.
This template is designed to clearly show various entity types with readable, descriptive text.
"""
from synthdoc.templates.base import BaseTemplate
from typing import Dict, Any
import random


class TestShowcaseTemplate(BaseTemplate):
    """Template for showcasing NER capabilities with clear, labeled test data"""

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        """Generate structured test data with REAL entity types from actual document templates"""
        if seed is not None:
            random.seed(seed)
            self.fake.seed_instance(seed)

        # Generate realistic legal entities
        plaintiff_name = self.fake.name()
        defendant_name = self.fake.company()
        attorney_first = self.fake.first_name()
        attorney_last = self.fake.last_name()
        attorney_name = f"{attorney_first} {attorney_last}, Esq."
        law_firm_name = f"{attorney_last} & Associates LLP"

        # Generate county and state for court
        county = self.fake.random_element(['Los Angeles', 'San Francisco', 'San Diego', 'Orange', 'Sacramento'])
        state = 'California'
        court_name = f"SUPERIOR COURT OF {state.upper()}, COUNTY OF {county.upper()}"

        # Generate case number
        case_year = random.randint(2024, 2026)
        case_num = random.randint(10000, 99999)
        case_number = f"CV-{case_year}-{case_num}"

        # Generate dates
        from datetime import datetime, timedelta
        date_filed = self.fake.date_between(start_date='-6m', end_date='today')
        statement_period_end = self.fake.date_between(start_date='-1m', end_date='today')
        statement_period_start = statement_period_end - timedelta(days=30)

        # Generate financial data
        bank_names = ['First National Bank', 'American Bank & Trust', 'City Financial Group', 'United Community Bank']
        bank_name = self.fake.random_element(bank_names)
        account_last_four = random.randint(1000, 9999)
        account_number = f"****{account_last_four}"

        # Generate balances with proper math
        beginning_balance = round(random.uniform(5000, 25000), 2)
        total_deposits = round(random.uniform(2000, 8000), 2)
        total_withdrawals = round(random.uniform(1500, 7000), 2)
        ending_balance = round(beginning_balance + total_deposits - total_withdrawals, 2)

        # Generate personal info
        person_name = self.fake.name()
        full_address = self.fake.address().replace('\n', ', ')
        phone_number = f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
        dob = self.fake.date_of_birth(minimum_age=18, maximum_age=85)

        # Generate medical entities
        patient_name = self.fake.name()
        diagnoses = ['Acute Bronchitis', 'Hypertension', 'Type 2 Diabetes', 'Migraine', 'Osteoarthritis']
        diagnosis = self.fake.random_element(diagnoses)
        provider_first = self.fake.first_name()
        provider_last = self.fake.last_name()
        provider_name = f"Dr. {provider_first} {provider_last}, MD"

        # Generate employment entities
        employee_name = self.fake.name()
        job_titles = ['Senior Software Engineer', 'Project Manager', 'Data Analyst', 'Marketing Director', 'Operations Manager']
        job_title = self.fake.random_element(job_titles)
        hourly_rate = round(random.uniform(45, 125), 2)
        salary_amount = f"${hourly_rate:.2f}/hr"
        company_name = self.fake.company()

        # Generate identifiers
        ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        ein = f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
        bar_number = random.randint(100000, 999999)
        license_number = f"CA-BAR-{bar_number}"

        # Generate notary/witness info
        notary_name = self.fake.name()
        notary_year = random.randint(2020, 2026)
        notary_num = random.randint(100000, 999999)
        notary_commission_number = f"NOT-CA-{notary_year}-{notary_num}"
        witness_name = self.fake.name()

        # Generate claim types
        claim_types = [
            'Breach of Contract', 'Fraud and Misrepresentation', 'Negligent Infliction of Emotional Distress',
            'Breach of Fiduciary Duty', 'Unjust Enrichment', 'Conversion', 'Defamation',
            'Professional Negligence', 'Intentional Interference with Contract', 'Violation of Civil Rights'
        ]
        selected_claims = random.sample(claim_types, 3)

        # Generate transaction data with running balance
        transaction_types = [
            ('DIRECT DEPOSIT - PAYROLL', 'credit', (2000, 5000)),
            ('ACH PAYMENT - RENT', 'debit', (1200, 2500)),
            ('DEBIT PURCHASE - GROCERIES', 'debit', (50, 300)),
            ('ATM WITHDRAWAL', 'debit', (20, 200)),
            ('ONLINE PAYMENT - ELECTRIC BILL', 'debit', (80, 250)),
            ('DEBIT PURCHASE - GAS STATION', 'debit', (30, 80)),
            ('CHECK - INSURANCE', 'debit', (150, 500)),
            ('DEBIT PURCHASE - RESTAURANT', 'debit', (25, 150)),
            ('ACH PAYMENT - CAR PAYMENT', 'debit', (300, 700)),
            ('DEBIT PURCHASE - ONLINE SHOPPING', 'debit', (40, 200)),
            ('WIRE TRANSFER IN', 'credit', (500, 3000)),
            ('MOBILE DEPOSIT', 'credit', (100, 1000)),
        ]

        transactions = []
        running_balance = beginning_balance
        current_date = statement_period_start
        ref_base = random.randint(780000, 999999)

        # Add beginning balance
        transactions.append({
            "date": current_date.strftime('%m/%d/%y'),
            "description": "BEGINNING BALANCE",
            "debit": "",
            "credit": "",
            "balance": f"${running_balance:,.2f}",
            "ref_code": f"REF-{ref_base}"
        })
        ref_base += 1

        # Generate 8-12 transactions
        num_transactions = random.randint(8, 12)
        for i in range(num_transactions):
            desc_template, trans_type, amount_range = random.choice(transaction_types)
            amount = round(random.uniform(*amount_range), 2)

            # Add some days
            current_date += timedelta(days=random.randint(1, 4))
            if current_date > statement_period_end:
                break

            if trans_type == 'credit':
                running_balance += amount
                transactions.append({
                    "date": current_date.strftime('%m/%d/%y'),
                    "description": desc_template,
                    "debit": "",
                    "credit": f"${amount:,.2f}",
                    "balance": f"${running_balance:,.2f}",
                    "ref_code": f"REF-{ref_base}"
                })
            else:
                running_balance -= amount
                # Add check number for checks
                if 'CHECK' in desc_template:
                    check_num = random.randint(1000, 9999)
                    desc = f"{desc_template} #{check_num}"
                else:
                    desc = desc_template
                transactions.append({
                    "date": current_date.strftime('%m/%d/%y'),
                    "description": desc,
                    "debit": f"${amount:,.2f}",
                    "credit": "",
                    "balance": f"${running_balance:,.2f}",
                    "ref_code": f"REF-{ref_base}"
                })
            ref_base += 1

        # Generate dates for document
        generation_date = datetime.now()
        signature_date_obj = self.fake.date_between(start_date='-30d', end_date='today')

        # Generate barcode and QR code
        barcode_num = random.randint(1000000000000, 9999999999999)
        qr_code_id = f"TEST_{random.randint(1000, 9999)}_{generation_date.year}"

        # Generate dynamic legal paragraphs using the generated names
        legal_paragraphs = [
            # PARTIES section
            f"Plaintiff {plaintiff_name} is an individual residing in {county} County, {state}, and is competent to testify to the matters set forth herein.",
            f"Defendant {defendant_name} is a Delaware corporation doing business in the State of {state} and maintains its principal place of business in {county} County.",
            f"Defendant conducts substantial business activities throughout the State of {state}, with offices in San Francisco, Los Angeles, and San Diego.",
            f"Plaintiff has been a resident of {state} for over twenty years and is sui juris.",
            f"At all times relevant to this Complaint, Defendant was engaged in the business of providing consulting services to individuals and businesses.",
            "Defendant held itself out to the public as a qualified and experienced consulting firm with expertise in business strategy and operations.",
            "The true names and capacities of defendants sued herein as DOES 1 through 50, inclusive, are unknown to Plaintiff at this time.",
            "Plaintiff is informed and believes, and on that basis alleges, that each of the DOE defendants is legally responsible for the events and happenings alleged herein.",
            "When the true identities and capacities of the DOE defendants are ascertained, Plaintiff will seek leave to amend this Complaint to assert their true names.",
            "Plaintiff is informed and believes that at all times mentioned herein, each defendant was the agent, servant, employee, or joint venturer of the other defendants.",

            # JURISDICTION section
            f"This Court has jurisdiction over this matter pursuant to {state} Code of Civil Procedure Section 410.10, as Defendant conducts substantial business activities within this State.",
            f"Venue is proper in {county} County because the events giving rise to this action occurred within this County and Defendant maintains offices here.",
            f"This Court has subject matter jurisdiction over all claims asserted herein pursuant to the {state} Constitution, Article VI, Section 10.",
            f"Personal jurisdiction over Defendant exists because Defendant transacts business in {state} and the claims arise out of Defendant's {state} activities.",
            f"Defendant has purposefully availed itself of the privilege of conducting activities in {state}, thereby invoking the benefits and protections of {state} law.",

            # FACTUAL BACKGROUND section
            f"On or about {date_filed.strftime('%B %d, %Y')}, Plaintiff and Defendant entered into a written contract whereby Defendant agreed to provide consulting services to Plaintiff.",
            "The contract was negotiated over a period of several weeks, with multiple meetings and exchanges of draft agreements between the parties.",
            "Plaintiff relied on Defendant's representations regarding its expertise, qualifications, and track record of success in similar engagements.",
            "Defendant represented that it had successfully completed over 500 similar consulting projects with a 98% client satisfaction rate.",
            f"Based on these representations, Plaintiff agreed to pay Defendant a total fee of ${random.randint(150000, 500000):,} for the consulting services.",
            "The contract required Defendant to perform a comprehensive analysis of Plaintiff's business operations and provide strategic recommendations.",
            "Defendant was to deliver a detailed written report within 90 days of the contract's execution, with interim progress reports every 30 days.",
            f"Plaintiff paid Defendant an initial retainer of ${random.randint(50000, 150000):,} upon execution of the contract, as required by the agreement.",
            "Defendant failed to provide the first interim progress report within the required 30-day timeframe.",
            "When Plaintiff inquired about the missing report, Defendant made excuses and promised delivery within one week.",

            # BREACH OF CONTRACT section
            "Defendant breached the contract by failing to perform the services as specified and misrepresenting its qualifications and capabilities.",
            "Defendant failed to conduct the comprehensive analysis required by the contract and instead provided only superficial and generic observations.",
            "The final report delivered by Defendant was plagiarized from publicly available materials and contained no original analysis.",
            "Defendant's employees lacked the qualifications and expertise that Defendant had represented they possessed.",
            "Defendant diverted resources to other clients during the engagement, in breach of its contractual obligations to Plaintiff.",
            "Defendant failed to meet numerous deadlines specified in the contract and failed to provide adequate explanations for the delays.",
            "Defendant's work product was of such poor quality that it provided no value to Plaintiff and could not be used for its intended purpose.",
            "Plaintiff provided Defendant with multiple opportunities to cure its breaches, but Defendant failed to do so.",

            # DAMAGES section
            f"As a direct and proximate result of Defendant's breaches, Plaintiff has suffered substantial damages including the ${random.randint(50000, 150000):,} retainer paid to Defendant.",
            f"Plaintiff has incurred additional costs of ${random.randint(50000, 100000):,} to retain a new consulting firm to perform the work that Defendant failed to complete.",
            f"Plaintiff has lost business opportunities valued at approximately ${random.randint(300000, 700000):,} due to delays caused by Defendant's failures.",
            "Plaintiff has suffered damage to its business reputation as a result of relying on Defendant's inadequate work product.",
            "Plaintiff has incurred attorneys' fees and costs in attempting to resolve this dispute informally before filing this action.",
        ]

        return {
            # Legal document entities
            "court_name": court_name,
            "case_number": case_number,
            "plaintiff_name": plaintiff_name,
            "defendant_name": defendant_name,
            "attorney_name": attorney_name,
            "law_firm": law_firm_name,
            "date_filed": date_filed.strftime('%m/%d/%Y'),
            "claim_type_1": selected_claims[0],
            "claim_type_2": selected_claims[1],
            "claim_type_3": selected_claims[2],

            # Financial entities
            "bank_name": bank_name,
            "account_number": account_number,
            "beginning_balance": f"${beginning_balance:,.2f}",
            "ending_balance": f"${ending_balance:,.2f}",
            "total_deposits": f"${total_deposits:,.2f}",
            "total_withdrawals": f"${total_withdrawals:,.2f}",
            "statement_period_start": statement_period_start.strftime('%m/%d/%Y'),
            "statement_period_end": statement_period_end.strftime('%m/%d/%Y'),

            # Personal/contact entities
            "person_name": person_name,
            "address": full_address,
            "phone_number": phone_number,
            "date_of_birth": dob.strftime('%m/%d/%Y'),

            # Medical entities
            "patient_name": patient_name,
            "diagnosis": diagnosis,
            "provider_name": provider_name,

            # Employment entities
            "employee_name": employee_name,
            "job_title": job_title,
            "salary_amount": salary_amount,
            "company_name": company_name,

            # Identifiers
            "ssn": ssn,
            "ein": ein,
            "license_number": license_number,

            # Notary information
            "notary_name": notary_name,
            "notary_commission_number": notary_commission_number,
            "witness_name": witness_name,

            # Transaction table data
            "transactions": transactions,

            # Legal paragraphs - dynamically generated
            "legal_paragraphs": legal_paragraphs,

            # Document metadata
            "document_title": "NER Comprehensive Test Showcase",
            "document_type": "COMPLAINT FOR DAMAGES",
            "generation_date": generation_date.strftime('%Y-%m-%d'),
            "signature_date": signature_date_obj.strftime('%B %d, %Y'),
            "version": "3.0",
            "barcode": f"||||| {barcode_num} |||||",
            "qr_code": f"[QR:{qr_code_id}]",
            "page_number": "Page 1 of N",
            "footer_text": "Test Showcase v3.0 - Real Entity Types from 95+ Document Templates"
        }


    def get_template_string(self) -> str:
        """Get template from external files (V2)."""
        html = self._load_file('template.html')
        css = self._load_file('styles.css')
        js = self._load_file('script.js')

        # Inject CSS into <head>
        if css and '</head>' in html:
            html = html.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)

        # Inject JS before </body>
        if js and '</body>' in html:
            html = html.replace('</body>', f'<script>\n{js}\n</script>\n</body>', 1)

        return html
    def get_name(self) -> str:
        return "test_showcase"

    def get_description(self) -> str:
        return "Test showcase template with REAL entity types from 95+ document templates - Legal, Financial, Medical, Employment"
