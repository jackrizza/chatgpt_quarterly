from lib.read.ingest import *
from lib.read.digest import *
from lib.read.ai import AI

import json

class JsonDataLoader:
    def __init__(self, json_file):
        self.data = self.load_json(json_file)
    
    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def labels(self):
        # Assuming the fields are stored in a list in the JSON file
        return self.data.get('labels', [])
    
    def get_train_data(self):
        # Assuming 'train_data' is a tuple in the format (string, dictionary)
        return self.data.get('train_data', (None, {}))
    

    def get_both(self):
        return self.labels(), self.get_train_data()
    

def train():

    labels, train_data = JsonDataLoader("training_weights_001.json").get_both()
    
    model = AI(labels=labels)
    model.train(train_data)
    model.save_model("model")


def main():

    labels = JsonDataLoader("training_weights_001.json").labels()

    ingest = Ingest("./test.pdf")
    ingest_text = ingest.get_text()
    print(ingest_text)
    digest = Digest().ai_approch(ingest_text, labels=labels, model="model")
    print(digest)
