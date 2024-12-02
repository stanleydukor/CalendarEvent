import json
import datetime
import argparse
import spacy
import dateutil
import datefinder

relative_words = {
    'today': 0,
    'tomorrow': 1,
    'next tomorrow': 2,
    'yesterday': -1,
    'day after tomorrow': 2,
    'day before yesterday': -2
}

class DateExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Downloading language model for the spaCy")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_date(self, text):
        doc = self.nlp(text)
        contains_date = False
        date = None

        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                contains_date = True
                break
        
        if contains_date:
            matches = list(datefinder.find_dates(text))
            if matches:
                date = matches[0]
            else:
                date = datetime.datetime.now().date()
            for word, offset in relative_words.items():
                if word in text:
                    date = date + datetime.timedelta(days=offset)
                    break
            return date
        
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dates from text")
    parser.add_argument("-p", "--path", type=str, help="Path to the event json file containing text data")
    args = parser.parse_args()

    output = {
        "date": None,
        "time": None
    }

    with open(args.path, "r") as file:
        data = json.load(file)
        cluster = data["lines"]

    extractor = DateExtractor()

    for line in cluster[::-1]:
        date = extractor.extract_date(line['message'])
        if output["date"] is None:
            try:
                output["date"] = date.date()
            except:
                pass
        if output["time"] is None:
            try:
                output["time"] = date.time()
            except:
                pass
        
    print(f"Date: {output['date']}, Time: {output['time']}")