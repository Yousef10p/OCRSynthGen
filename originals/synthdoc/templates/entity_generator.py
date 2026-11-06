"""
Entity generator for consistent, realistic synthetic data across documents.
Provides coherent PERSON, ORG, DATE, ADDRESS, AMOUNT, PHONE, ID, EMAIL, LOCATION, QUANTITY, URL, DURATION entities.
"""
import random
import string
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from faker import Faker


@dataclass
class Person:
    first_name: str
    last_name: str
    middle_name: str = ""
    suffix: str = ""
    gender: str = ""
    dob: datetime = None
    ssn: str = ""
    email: str = ""
    phone: str = ""
    address: 'Address' = None
    
    @property
    def full_name(self) -> str:
        parts = [self.first_name]
        if self.middle_name:
            parts.append(self.middle_name[0] + ".")
        parts.append(self.last_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)
    
    @property
    def formal_name(self) -> str:
        return f"{self.last_name}, {self.first_name}" + (f" {self.middle_name[0]}." if self.middle_name else "")


@dataclass
class Organization:
    name: str
    dba_name: str = ""
    ein: str = ""
    phone: str = ""
    email: str = ""
    website: str = ""
    address: 'Address' = None
    industry: str = ""
    
    @property
    def display_name(self) -> str:
        return self.dba_name if self.dba_name else self.name


@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    street2: str = ""
    state_full: str = ""
    country: str = "USA"
    
    @property
    def one_line(self) -> str:
        parts = [self.street]
        if self.street2:
            parts.append(self.street2)
        parts.append(f"{self.city}, {self.state} {self.zip_code}")
        return ", ".join(parts)
    
    @property
    def multi_line(self) -> str:
        lines = [self.street]
        if self.street2:
            lines.append(self.street2)
        lines.append(f"{self.city}, {self.state} {self.zip_code}")
        return "\n".join(lines)


STATE_ABBR_TO_FULL = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

STATE_ZIP_RANGES = {
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
    "WI": (53001, 54990), "WY": (82001, 83128), "DC": (20001, 20099),
}

INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Manufacturing", "Retail",
    "Education", "Construction", "Legal Services", "Real Estate", "Insurance",
    "Transportation", "Energy", "Telecommunications", "Agriculture", "Hospitality",
]

SUFFIXES = ["", "", "", "", "Jr.", "Sr.", "II", "III", "IV"]


class EntityGenerator:
    """Generates coherent, realistic entities for synthetic documents."""
    
    def __init__(self, seed: int = None):
        self.fake = Faker()
        if seed is not None:
            random.seed(seed)
            self.fake.seed_instance(seed)
        
        self._cache: Dict[str, Any] = {}
        self._persons: Dict[str, Person] = {}
        self._organizations: Dict[str, Organization] = {}
    
    def _get_state_zip(self, state_abbr: str = None) -> Tuple[str, str]:
        if state_abbr is None:
            state_abbr = random.choice(list(STATE_ZIP_RANGES.keys()))
        zip_lo, zip_hi = STATE_ZIP_RANGES[state_abbr]
        zip_code = str(random.randint(zip_lo, zip_hi)).zfill(5)
        return state_abbr, zip_code
    
    def generate_address(self, state: str = None, key: str = None) -> Address:
        if key and key in self._cache:
            return self._cache[key]
        
        state_abbr, zip_code = self._get_state_zip(state)
        addr = Address(
            street=self.fake.street_address(),
            street2=self.fake.secondary_address() if random.random() < 0.2 else "",
            city=self.fake.city(),
            state=state_abbr,
            state_full=STATE_ABBR_TO_FULL.get(state_abbr, state_abbr),
            zip_code=zip_code,
        )
        if key:
            self._cache[key] = addr
        return addr
    
    def generate_person(
        self, 
        key: str = None,
        min_age: int = 18, 
        max_age: int = 85,
        gender: str = None,
        with_email: bool = True,
        with_phone: bool = True,
        with_address: bool = True,
        with_ssn: bool = False,
    ) -> Person:
        if key and key in self._persons:
            return self._persons[key]
        
        gender = gender or random.choice(["Male", "Female"])
        if gender == "Male":
            first_name = self.fake.first_name_male()
            middle_name = self.fake.first_name_male() if random.random() < 0.6 else ""
        else:
            first_name = self.fake.first_name_female()
            middle_name = self.fake.first_name_female() if random.random() < 0.6 else ""
        
        last_name = self.fake.last_name()
        suffix = random.choice(SUFFIXES)
        
        dob = self.fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)
        
        email = ""
        if with_email:
            email_user = f"{first_name.lower()}.{last_name.lower()}"
            email_domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"])
            email = f"{email_user}@{email_domain}"
        
        phone = ""
        if with_phone:
            phone = self._generate_phone()
        
        ssn = ""
        if with_ssn:
            ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        
        person = Person(
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name,
            suffix=suffix,
            gender=gender,
            dob=dob,
            ssn=ssn,
            email=email,
            phone=phone,
            address=self.generate_address() if with_address else None,
        )
        
        if key:
            self._persons[key] = person
        return person
    
    def generate_organization(
        self, 
        key: str = None,
        industry: str = None,
        with_dba: bool = False,
        with_ein: bool = True,
        with_contact: bool = True,
    ) -> Organization:
        if key and key in self._organizations:
            return self._organizations[key]
        
        name = self.fake.company()
        dba_name = self.fake.company() if with_dba and random.random() < 0.3 else ""
        industry = industry or random.choice(INDUSTRIES)
        
        ein = ""
        if with_ein:
            ein = f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
        
        phone = ""
        email = ""
        website = ""
        if with_contact:
            phone = self._generate_phone()
            domain = name.lower().replace(" ", "").replace(",", "").replace(".", "").replace("&", "")[:15]
            domain = ''.join(c for c in domain if c.isalnum())
            email = f"info@{domain}.com"
            website = f"www.{domain}.com"
        
        org = Organization(
            name=name,
            dba_name=dba_name,
            ein=ein,
            phone=phone,
            email=email,
            website=website,
            address=self.generate_address(),
            industry=industry,
        )
        
        if key:
            self._organizations[key] = org
        return org
    
    def _generate_phone(self) -> str:
        area = random.randint(200, 999)
        exchange = random.randint(200, 999)
        subscriber = random.randint(1000, 9999)
        return f"({area}) {exchange}-{subscriber}"
    
    def generate_date(
        self,
        start: datetime = None,
        end: datetime = None,
        future_days: int = None,
        past_days: int = None,
    ) -> datetime:
        if future_days:
            return self.fake.future_date(end_date=f"+{future_days}d")
        if past_days:
            return self.fake.past_date(start_date=f"-{past_days}d")
        if start and end:
            return self.fake.date_between(start_date=start, end_date=end)
        return self.fake.date_this_year()
    
    def generate_duration(self, min_days: int = 1, max_days: int = 365) -> timedelta:
        return timedelta(days=random.randint(min_days, max_days))
    
    def generate_amount(
        self, 
        min_val: float = 0.01, 
        max_val: float = 10000.00,
        as_string: bool = True,
        with_currency: bool = True,
    ) -> str | float:
        amount = round(random.uniform(min_val, max_val), 2)
        if as_string:
            return f"${amount:,.2f}" if with_currency else f"{amount:,.2f}"
        return amount
    
    def generate_id(
        self,
        prefix: str = "",
        length: int = 8,
        alphanumeric: bool = False,
    ) -> str:
        if alphanumeric:
            chars = string.ascii_uppercase + string.digits
            body = ''.join(random.choice(chars) for _ in range(length))
        else:
            body = ''.join(random.choice(string.digits) for _ in range(length))
        return f"{prefix}{body}" if prefix else body
    
    def generate_quantity(self, min_val: int = 1, max_val: int = 100) -> int:
        return random.randint(min_val, max_val)
    
    def generate_url(self, domain: str = None) -> str:
        if domain:
            path = self.fake.slug()
            return f"https://www.{domain}/{path}"
        return self.fake.url()
    
    def get_cached_person(self, key: str) -> Optional[Person]:
        return self._persons.get(key)
    
    def get_cached_org(self, key: str) -> Optional[Organization]:
        return self._organizations.get(key)
    
    def clear_cache(self):
        self._cache.clear()
        self._persons.clear()
        self._organizations.clear()
    
    def format_date(self, date: datetime, fmt: str = "%m/%d/%Y") -> str:
        return date.strftime(fmt)
    
    def format_amount(self, amount: float, with_currency: bool = True) -> str:
        return f"${amount:,.2f}" if with_currency else f"{amount:,.2f}"
    
    def calculate_age(self, dob: datetime, as_of: datetime = None) -> int:
        as_of = as_of or datetime.now()
        return as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))


entity_generator = EntityGenerator()
