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
    
    def __init__(self) :
        pass

    def ai_approch(self, text: str, labels=None, model=None):
        if model is None or labels is None:
            raise ValueError("Model or Labels not provided.")

        model = AI(model=model, labels=labels)
        predictions = model.predict(text)

        for prediction in predictions:
            label, value = prediction
            if label == "date of filing":
                self.date_of_filing = value
            elif label == "company name":
                self.company_name = value
            elif label == "quarter":
                self.quarter = value
            elif label == "year":
                self.year = value
            elif label == "revenue":
                self.revenue = float(value)
            elif label == "net income":
                self.net_income = float(value)
            elif label == "intrest":
                self.intrest = float(value)
            elif label == "taxes":
                self.taxes = float(value)
            elif label == "deprecation":
                self.deprecation = float(value)
            elif label == "amortization":
                self.amortization = float(value)
            elif label == "eps":
                self.eps = float(value)

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
