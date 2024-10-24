import spacy
from spacy.training import Example
from spacy.util import minibatch
import random


class AI:
    def __init__(self, labels, model="en_core_web_sm", iterations=10):
        """
        Initialize the SpacyNERTrainer class.

        :param labels: List of labels for the NER model (e.g., ["EBITDA", "EPS", "OUTLOOK"]).
        :param model: Pre-trained model to load (default is "en_core_web_sm").
        :param iterations: Number of training iterations (default is 10).
        """
        self.labels = labels
        self.iterations = iterations
        self.nlp = spacy.load(model)  # Load a pre-trained spaCy model

        # Add 'ner' pipe to the pipeline if it's not present
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        # Add new labels to the NER pipeline
        for label in labels:
            ner.add_label(label)

    def train(self, train_data):
        """
        Train the model using the provided training data.

        :param train_data: A list of tuples with texts and their annotations.
                           Example: [("EBITDA was $2B", {"entities": [(0, 6, "EBITDA")]}), ...]
        """
        # Disable other pipeline components during training, only keep 'ner' active
        pipe_exceptions = ["ner"]
        unaffected_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        # Prepare training
        optimizer = self.nlp.resume_training()

        with self.nlp.disable_pipes(*unaffected_pipes):  # Train only NER
            for iteration in range(self.iterations):
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=8)
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)

                    self.nlp.update(examples, sgd=optimizer, losses=losses)

                print(f"Iteration {iteration+1} - Losses: {losses}")

    def predict(self, text):
        """
        Use the trained model to make predictions on new text.

        :param text: The input text to perform NER on.
        :return: A list of tuples with the entity label and text.
        """
        doc = self.nlp(text)
        predictions = [(ent.text, ent.label_) for ent in doc.ents]
        return predictions

    def save_model(self, output_dir):
        """
        Save the trained model to the specified directory.

        :param output_dir: Directory to save the model.
        """
        self.nlp.to_disk(output_dir)
        print(f"Model saved to {output_dir}")

    def load_model(self, model_dir):
        """
        Load a trained model from the specified directory.

        :param model_dir: Directory to load the model from.
        """
        self.nlp = spacy.load(model_dir)
        print(f"Model loaded from {model_dir}")
