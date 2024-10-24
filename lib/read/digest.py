from lib.read.ai import AI
from dataclasses import dataclass

import re


def extract_value(text: str, key: str):
    pattern = re.compile(rf"{key}:\s+\$?(\d+.\d+)")
    match = pattern.search(text)
    return match.group(1) if match else None


@dataclass
class Digest:

    # These values have to be spot on to be used as keys in the dictionary
    date_of_filing: str
    company_name: str
    quarter: str
    year: str

    # these will expand and can be null
    revenue: float
    net_income: float
    intrest: float
    taxes: float
    deprecation: float
    amortization: float
    eps: float

    def __init__(self):
        self.date_of_filing = None
        self.company_name = None
        self.quarter = None
        self.year = None

        self.revenue = None
        self.net_income = None
        self.intrest = None
        self.taxes = None
        self.deprecation = None
        self.amortization = None
        self.eps = None

    def ai_approch(self, text: str, labels=None, model=None):
        if model is None or labels is None:
            raise ValueError("Model or Labels not provided.")
        
        ai = AI(labels=labels, model=model)
        dictionary = ai.classify(text)
        self.date_of_filing = dictionary.get("date of filing")
        self.company_name = dictionary.get("company name")
        self.quarter = dictionary.get("quarter")
        self.year = dictionary.get("year")
        
        self.revenue = dictionary.get("revenue")
        self.net_income = dictionary.get("net income")
        self.intrest = dictionary.get("intrest")
        self.taxes = dictionary.get("taxes")
        self.deprecation = dictionary.get("deprecation")
        self.amortization = dictionary.get("amortization")
        self.eps = dictionary.get("eps")


    def regex_approch(self, text: str):

        self.date_of_filing = extract_value(text, "date of filing")
        self.company_name = extract_value(text, "company name")
        self.quarter = extract_value(text, "quarter")
        self.year = extract_value(text, "year")

        self.revenue = extract_value(text, "revenue")
        self.net_income = extract_value(text, "net income")
        self.intrest = extract_value(text, "intrest")
        self.taxes = extract_value(text, "taxes")
        self.deprecation = extract_value(text, "deprecation")
        self.amortization = extract_value(text, "amortization")
        self.eps = extract_value(text, "eps")
