"""
BankStatementTemplate implementation using the new template system.
"""
from synthdoc.templates.base import BaseTemplate
from typing import Dict, Any
import random
from datetime import datetime, timedelta


class BankStatementTemplate(BaseTemplate):
    """Template for financial bank statements"""

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        """Generate random but realistic bank statement data"""
        if seed is not None:
            random.seed(seed)
            self.fake.seed_instance(seed)

        bank_name_to_domain = {
            "FIRST NATIONAL BANK": "firstnational.com",
            "AMERICAN BANK & TRUST": "ambanktrust.com",
            "CITY FINANCIAL GROUP": "cityfinancial.com",
            "GLOBAL TRUST BANK": "globaltrust.com",
            "METROPOLITAN CREDIT UNION": "metrocu.org",
            "PREMIER SAVINGS BANK": "premiersavings.com",
            "NATIONAL COMMERCE BANK": "natcommerce.com",
            "FEDERAL SAVINGS & LOAN": "fedsavings.com",
            "HERITAGE COMMUNITY BANK": "heritagebank.com",
            "PACIFIC WEST BANKING": "pacwestbanking.com",
            "CITIZENS STATE BANK": "citizensstate.com",
            "LIBERTY FEDERAL CREDIT UNION": "libertyfcu.org",
            "SUMMIT FINANCIAL GROUP": "summitfinancial.com",
            "CORNERSTONE BANK": "cornerstonebank.com",
            "VALLEY NATIONAL BANK": "valleynational.com",
            "RIVERSIDE COMMUNITY BANK": "riversidebank.com",
            "PIONEER TRUST BANK": "pioneertrust.com",
            "UNITED MIDWEST BANK": "unitedmidwest.com",
            "EAGLE SAVINGS BANK": "eaglesavings.com",
            "HARBOR FEDERAL CREDIT UNION": "harborfcu.org",
            "CRESTVIEW BANK & TRUST": "crestviewbt.com",
            "CONTINENTAL SAVINGS BANK": "continentalsavings.com",
            "FRONTIER STATE BANK": "frontierstate.com",
            "SILVERADO FINANCIAL": "silveradofinancial.com",
            "BEACON HILL BANK": "beaconhillbank.com",
        }
        bank_names = list(bank_name_to_domain.keys())

        deposit_descriptions = [
            ("DIRECT DEPOSIT", 25),
            ("CHECK DEPOSIT", 15),
            ("CASH DEPOSIT", 10),
            ("INTEREST CREDIT", 5),
            ("ONLINE TRANSFER", 15),
            ("WIRE TRANSFER IN", 10),
            ("REFUND", 10),
            ("PAYROLL", 10),
        ]

        withdrawal_descriptions = [
            ("ATM WITHDRAWAL", 15),
            ("ONLINE PURCHASE", 15),
            ("RECURRING PAYMENT", 10),
            ("BANK FEE", 2),
            ("LOAN PAYMENT", 10),
            ("DEBIT CARD PURCHASE", 20),
            ("BILL PAY", 15),
            ("WIRE TRANSFER OUT", 8),
        ]

        merchant_names = [
            "WALMART", "TARGET", "AMAZON.COM", "COSTCO",
            "WALGREENS", "CVS PHARMACY", "HOME DEPOT", "LOWES",
            "SHELL OIL", "CHEVRON", "STARBUCKS", "MCDONALD'S",
            "NETFLIX.COM", "SPOTIFY USA", "AT&T", "VERIZON",
            "WHOLE FOODS", "TRADER JOES", "KROGER", "PUBLIX",
            "UBER TRIP", "LYFT RIDE", "DOORDASH", "GRUBHUB",
        ]

        bill_payees = [
            "ELECTRIC CO", "GAS UTILITY", "WATER DEPT",
            "CABLE/INTERNET", "INSURANCE CO", "PHONE BILL",
            "CREDIT CARD", "STUDENT LOAN", "AUTO INSURANCE",
            "PROPERTY TAX", "HOA DUES", "TRASH SERVICE",
        ]

        state_zip_ranges = {
            "AL": (35004, 36925), "AK": (99501, 99950), "AZ": (85001, 86556),
            "AR": (71601, 72959), "CA": (90001, 96162), "CO": (80001, 81658),
            "CT": (6001, 6928), "DE": (19701, 19980), "FL": (32003, 34997),
            "GA": (30002, 31999), "HI": (96701, 96898), "ID": (83201, 83877),
            "IL": (60001, 62999), "IN": (46001, 47997), "IA": (50001, 52809),
            "KS": (66002, 67954), "KY": (40003, 42788), "LA": (70001, 71497),
            "ME": (3901, 4992), "MD": (20601, 21930), "MA": (1001, 2791),
            "MI": (48001, 49971), "MN": (55001, 56763), "MS": (38601, 39776),
            "MO": (63001, 65899), "MT": (59001, 59937), "NE": (68001, 69367),
            "NV": (88901, 89883), "NH": (3031, 3897), "NJ": (7001, 8989),
            "NM": (87001, 88439), "NY": (10001, 14925), "NC": (27006, 28909),
            "ND": (58001, 58856), "OH": (43001, 45999), "OK": (73001, 74966),
            "OR": (97001, 97920), "PA": (15001, 19640), "RI": (2801, 2940),
            "SC": (29001, 29945), "SD": (57001, 57799), "TN": (37010, 38589),
            "TX": (73301, 79999), "UT": (84001, 84791), "VT": (5001, 5907),
            "VA": (20101, 24658), "WA": (98001, 99403), "WV": (24701, 26886),
            "WI": (53001, 54990), "WY": (82001, 83128),
        }
        us_states = list(state_zip_ranges.keys())

        account_type = self.fake.random_element(["Checking", "Savings"])

        end_date = self.fake.date_between(start_date='-6m', end_date='today')
        period_days = self.fake.random_int(min=28, max=31)
        start_date = end_date - timedelta(days=period_days)

        num_transactions = self.fake.random_int(min=8, max=13)
        transactions = []
        beginning_balance = round(self.fake.random.uniform(1500.00, 12000.00), 2)
        running_balance = beginning_balance

        tx_dates = sorted([
            self.fake.date_between(start_date=start_date, end_date=end_date)
            for _ in range(num_transactions)
        ])

        def weighted_choice(items_with_weights):
            items, weights = zip(*items_with_weights)
            return random.choices(items, weights=weights, k=1)[0]

        used_interest_credit = False
        used_bank_fee = False
        used_payroll = False
        direct_deposit_count = 0

        for i, transaction_date in enumerate(tx_dates):
            if running_balance < 500:
                is_deposit = True
            elif running_balance > 15000:
                is_deposit = False
            else:
                is_deposit = self.fake.boolean(chance_of_getting_true=35)

            if is_deposit:
                description = weighted_choice(deposit_descriptions)
                if description == "INTEREST CREDIT" and used_interest_credit:
                    description = weighted_choice([
                        d for d in deposit_descriptions if d[0] != "INTEREST CREDIT"
                    ])
                if description == "PAYROLL" and used_payroll:
                    description = weighted_choice([
                        d for d in deposit_descriptions if d[0] not in ("PAYROLL", "INTEREST CREDIT")
                    ])
                if description == "DIRECT DEPOSIT" and direct_deposit_count >= 2:
                    description = weighted_choice([
                        d for d in deposit_descriptions if d[0] not in ("DIRECT DEPOSIT", "INTEREST CREDIT", "PAYROLL")
                    ])
                if description == "INTEREST CREDIT":
                    used_interest_credit = True
                    amount = round(self.fake.random.uniform(1.50, 85.00), 2)
                elif description == "PAYROLL":
                    used_payroll = True
                    amount = round(self.fake.random.uniform(800.00, 4500.00), 2)
                elif description == "DIRECT DEPOSIT":
                    direct_deposit_count += 1
                    amount = round(self.fake.random.uniform(800.00, 4500.00), 2)
                elif description == "WIRE TRANSFER IN":
                    amount = round(self.fake.random.uniform(500.00, 5000.00), 2)
                else:
                    amount = round(self.fake.random.uniform(50.00, 3500.00), 2)
                debit_amt = ""
                credit_amt = f"${amount:,.2f}"
                running_balance = round(running_balance + amount, 2)
            else:
                description = weighted_choice(withdrawal_descriptions)
                if description == "BANK FEE" and used_bank_fee:
                    description = weighted_choice([
                        d for d in withdrawal_descriptions if d[0] != "BANK FEE"
                    ])
                if description == "BANK FEE":
                    used_bank_fee = True
                max_withdrawal = min(running_balance - 50, 3000.00)
                if max_withdrawal < 5:
                    continue
                if description == "BANK FEE":
                    amount = round(self.fake.random.uniform(5.00, 45.00), 2)
                elif description == "ATM WITHDRAWAL":
                    amount = round(self.fake.random.uniform(20.00, 500.00), 2)
                elif description == "LOAN PAYMENT":
                    amount = round(self.fake.random.uniform(150.00, 1800.00), 2)
                else:
                    amount = round(self.fake.random.uniform(5.00, max_withdrawal), 2)
                amount = min(amount, max_withdrawal)
                debit_amt = f"${amount:,.2f}"
                credit_amt = ""
                running_balance = round(running_balance - amount, 2)

            is_merchant = False
            if description in ("DEBIT CARD PURCHASE", "ONLINE PURCHASE", "RECURRING PAYMENT"):
                merchant = self.fake.random_element(merchant_names)
                store_num = self.fake.random_int(min=100, max=9999)
                state_abbr = self.fake.random_element(us_states)
                description = f"{merchant} #{store_num} {state_abbr}"
                is_merchant = True
            elif description == "BILL PAY":
                payee = self.fake.random_element(bill_payees)
                description = f"BILL PAY - {payee}"
                is_merchant = True

            transaction = {
                "date": transaction_date.strftime("%m/%d/%Y"),
                "description": description,
                "is_merchant": is_merchant,
                "ref_code": f"REF{self.fake.random_int(100000, 999999)}",
                "debit_amt": debit_amt,
                "credit_amt": credit_amt,
                "running_bal": f"${running_balance:,.2f}"
            }
            transactions.append(transaction)

        # Assign categories to transactions
        categories_list = [
            "Shopping", "Dining", "Transport", "Utilities",
            "Healthcare", "Entertainment", "Income", "Fees", "Transfer"
        ]
        deposit_categories = ["Income", "Transfer"]
        withdrawal_category_pool = [
            "Shopping", "Dining", "Transport", "Utilities",
            "Healthcare", "Entertainment", "Fees"
        ]

        for t in transactions:
            if t['credit_amt']:
                t['category'] = self.fake.random_element(deposit_categories)
            else:
                desc_upper = t['description'].upper()
                if "BILL PAY" in desc_upper or "ELECTRIC" in desc_upper or "GAS" in desc_upper or "WATER" in desc_upper:
                    t['category'] = "Utilities"
                elif "ATM" in desc_upper or "BANK FEE" in desc_upper:
                    t['category'] = "Fees"
                elif "STARBUCKS" in desc_upper or "MCDONALD" in desc_upper or "DOORDASH" in desc_upper or "GRUBHUB" in desc_upper:
                    t['category'] = "Dining"
                elif "UBER" in desc_upper or "LYFT" in desc_upper or "SHELL" in desc_upper or "CHEVRON" in desc_upper:
                    t['category'] = "Transport"
                elif "NETFLIX" in desc_upper or "SPOTIFY" in desc_upper:
                    t['category'] = "Entertainment"
                else:
                    t['category'] = self.fake.random_element(withdrawal_category_pool)

        total_deposits = sum(
            float(t['credit_amt'].replace('$', '').replace(',', ''))
            for t in transactions if t['credit_amt']
        )
        total_withdrawals = sum(
            float(t['debit_amt'].replace('$', '').replace(',', ''))
            for t in transactions if t['debit_amt']
        )

        # Build category summary from withdrawal transactions
        category_totals = {}
        for t in transactions:
            if t['debit_amt']:
                amt = float(t['debit_amt'].replace('$', '').replace(',', ''))
                cat = t['category']
                category_totals[cat] = category_totals.get(cat, 0) + amt

        category_summary = []
        if total_withdrawals > 0:
            for cat, amt in sorted(category_totals.items(), key=lambda x: -x[1]):
                pct = round((amt / total_withdrawals) * 100)
                category_summary.append({
                    "name": cat,
                    "amount": f"${amt:,.2f}",
                    "percentage": pct
                })

        area_code = self.fake.random_element(["800", "888", "877", "866"])
        cs_phone = f"1-{area_code}-{self.fake.random_int(200, 999)}-{self.fake.random_int(1000, 9999)}"
        selected_bank_name = self.fake.random_element(bank_names)
        bank_domain = bank_name_to_domain[selected_bank_name]
        cs_email = f"support@{bank_domain}"

        account_holder = f"{self.fake.first_name()} {self.fake.last_name()}"

        bank_state = self.fake.random_element(us_states)
        zip_lo, zip_hi = state_zip_ranges[bank_state]
        bank_zip = str(self.fake.random_int(min=zip_lo, max=zip_hi)).zfill(5)

        generated_date = datetime.now().strftime("%m/%d/%Y")
        statement_id = f"STMT-{self.fake.random_int(100000, 999999)}"
        routing_number = f"{self.fake.random_int(100000000, 999999999):09d}"
        bank_website = f"www.{bank_domain}"
        service_hours = self.fake.random_element([
            "Monday-Friday 8:00 AM - 6:00 PM EST",
            "Monday-Friday 9:00 AM - 5:00 PM EST",
            "24/7 Customer Service Available",
            "Monday-Saturday 8:00 AM - 8:00 PM EST"
        ])

        return {
            "bank_name": selected_bank_name,
            "bank_address": f"{self.fake.street_address()}, {self.fake.city()}, {bank_state} {bank_zip}",
            "account_holder": account_holder,
            "account_number": f"****{self.fake.random_int(1000, 9999)}",
            "account_type": account_type,
            "start_date": start_date.strftime("%m/%d/%Y"),
            "end_date": end_date.strftime("%m/%d/%Y"),
            "beginning_balance": f"${beginning_balance:,.2f}",
            "ending_balance": f"${running_balance:,.2f}",
            "total_deposits": f"${total_deposits:,.2f}",
            "total_withdrawals": f"${total_withdrawals:,.2f}",
            "transactions": transactions,
            "category_summary": category_summary,
            "customer_service_phone": cs_phone,
            "customer_service_email": cs_email,
            "bank_website": bank_website,
            "generated_date": generated_date,
            "routing_number": routing_number,
            "service_hours": service_hours,
            "statement_id": statement_id,
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
