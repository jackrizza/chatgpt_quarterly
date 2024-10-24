from lib.read.ingest import *
from lib.read.digest import *
from lib.read.ai import AI


labels = ["revenue", "net income", "intrest", "taxes", "deprecation", "eps"]
train_data = [
    (
        "The company reported a revenue of $1B in the last quarter.",
        {"entities": [(37, 40, "revenue")]},
    ),
    ("The net income was $5B.", {"entities": [(20, 23, "net income")]}),
    ("The company had an intrest of $2B.", {"entities": [(26, 29, "intrest")]}),
    ("The taxes paid were $1B.", {"entities": [(19, 22, "taxes")]}),
    ("The deprecation was $3B.", {"entities": [(19, 22, "deprecation")]}),
    ("The company reported an EPS of $2.", {"entities": [(31, 32, "eps")]}),
]


def train():

    model = AI(labels=labels)
    model.train(train_data)
    model.save_model("model")


def main():
    ingest = Ingest("./test.pdf")
    ingest_text = ingest.get_text()
    digest = Digest().ai_approch(ingest_text, labels=labels, model="model")
    print(digest)
