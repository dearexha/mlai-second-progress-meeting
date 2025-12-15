import nltk
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel

# nltk.download('punkt')
# nltk.download('punkt_tab')

def sentence_stream_generator(filepath, label):
    """
    This generator reads a file, splits lines into sentences,
    and yields one {'text': sentence, 'label': label} dict
    for each sentence.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split("\t", maxsplit=2)

            # each line contains multiple sentences, and has the format article_title<TAB>paragraph_number<TAB>sentence
            sentences = nltk.sent_tokenize(parts[2])

            # yield sentence and label
            for sent in sentences:
                if sent.strip():
                    yield {
                        "text": sent,
                        "label": label
                    }


if __name__=="__main__":
    print("Loading raw dataset ...")
    # paths to SL and SL data
    file_label_map = {
        0: "./datasets/SimpleWikipedia_v2/simple.aligned",
        1: "./datasets/SimpleWikipedia_v2/normal.aligned"
    }

    # create a list of huggingface datasets
    labeled_datasets_list = []
    for label, filepath in file_label_map.items():

        # create HF dataset from generator
        ds = Dataset.from_generator(
            sentence_stream_generator,
            gen_kwargs={
                "filepath": filepath,
                "label": label
            }
        )
        labeled_datasets_list.append(ds)

    # concatenate datasets for boths classes
    raw_dataset = concatenate_datasets(labeled_datasets_list)

    label_names = ["SL", "EL"]
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=label_names)
    })
    raw_dataset = raw_dataset.cast(features)

    print("Loading raw dataset done")
    print("Saving as HF dataset...")
    raw_dataset.save_to_disk("./results/SimpleWikipedia_raw")  # only contains labels and sentences