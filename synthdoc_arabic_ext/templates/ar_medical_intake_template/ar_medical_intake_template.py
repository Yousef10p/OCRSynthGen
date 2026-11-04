"""
Arabic patient intake form (نموذج معلومات المريض), from the medical-intake section
of SynthDoc's ``test_showcase`` template: a grid of labelled fields, insurance
block, free-text boxes, checkboxes and a consent paragraph.
"""
from typing import Any, Dict

from synthdoc.templates.test_showcase_template.test_showcase_template import TestShowcaseTemplate
from synthdoc_arabic.base import ArabicTemplateMixin
from synthdoc_arabic.data import (ALLERGIES, CHRONIC_CONDITIONS, DOCTOR_SPECIALTIES, INSURERS,
                                  MEDICATIONS, NATIONALITIES, NATIONALITIES_F, RELATIONSHIPS,
                                  VISIT_REASONS)

HOSPITALS = ["مستشفى الرعاية التخصصي", "مجمع الشفاء الطبي", "مستشفى الأمل العام", "مركز الحياة الطبي"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]


class ArabicMedicalIntakeTemplate(ArabicTemplateMixin, TestShowcaseTemplate):
    """نموذج معلومات المريض - Arabic patient intake form"""

    BASE_TEMPLATE = TestShowcaseTemplate
    FONT_CHOICES = ["noto_naskh", "ibm_plex", "tajawal"]
    PAGE_SIZES = ["A4"]

    def get_archetype(self) -> str:
        return "form"

    def get_category(self) -> str:
        return "medical"

    def generate_sample_data(self, seed: int = None) -> Dict[str, Any]:
        ar = self.arabic_setup(seed)
        patient = ar.person()
        emergency = ar.person()
        doctor = ar.person()
        visit = ar.recent_date(60)
        birth = ar.date_between(visit.replace(year=visit.year - 80), visit.replace(year=visit.year - 18))
        age = visit.year - birth.year - ((visit.month, visit.day) < (birth.month, birth.day))
        nationality = ar.choice(NATIONALITIES)
        conditions = ar.sample(CHRONIC_CONDITIONS, ar.rng.randint(0, 2))
        hospital = ar.choice(HOSPITALS)

        return {
            "hospital_name": hospital,
            "department": ar.choice(["العيادات الخارجية", "قسم الطوارئ", "عيادة طب الأسرة"]),
            "file_number": f"MRN-{ar.rng.randint(100000, 999999)}",
            "visit_date": ar.date_any(visit),
            "family_name": patient["family"],
            "first_name": f"{patient['first']} {patient['father']}",
            "date_of_birth": ar.date_numeric(birth),
            "age": ar.digits(age),
            "gender": "ذكر" if patient["gender"] == "m" else "أنثى",
            "nationality": nationality if patient["gender"] == "m" else NATIONALITIES_F[nationality],
            "id_label": ar.country["id_label"],
            "national_id": ar.national_id(),
            "blood_type": ar.choice(BLOOD_TYPES),
            "address": ar.address(),
            "phone_number": ar.phone(),
            "email": ar.email(person=patient),
            "emergency_name": emergency["full"],
            "emergency_relationship": ar.choice(RELATIONSHIPS),
            "emergency_phone": ar.phone(),
            "insurance_provider": ar.choice(INSURERS),
            "policy_number": f"POL-{ar.rng.randint(10 ** 7, 10 ** 8 - 1)}",
            "member_class": ar.choice(["الفئة الذهبية (VIP)", "الفئة أ", "الفئة ب", "الفئة ج"]),
            "expiry_date": ar.date_numeric(visit.replace(year=visit.year + 1)),
            "physician": f"د. {doctor['full']}",
            "specialty": ar.choice(DOCTOR_SPECIALTIES),
            "medications": "، ".join(ar.sample(MEDICATIONS, ar.rng.randint(1, 3))),
            "allergies": ar.choice(ALLERGIES),
            "conditions": [{"name": c, "checked": c in conditions} for c in CHRONIC_CONDITIONS],
            "visit_reason": ar.choice(VISIT_REASONS),
            "patient_full_name": patient["full"],
            "signature_date": ar.date_numeric(visit),
        }
