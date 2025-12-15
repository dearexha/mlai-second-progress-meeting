import nltk
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel
from pathlib import Path

# nltk.download('punkt')
# nltk.download('punkt_tab')

def sentence_stream_generator(filepath, label):
    """
    This generator reads a file, splits lines into sentences,
    and yields one {'text': sentence, 'label': label} dict
    for each sentence.
    """
    scan_dir = Path(filepath)
    if scan_dir.is_dir():
        for item in scan_dir.iterdir():
            if item.is_file():
                try:
                    # each paragraph contains multiple sentences
                    text = item.read_text(encoding="utf-8")

                    if not text.strip():
                        continue

                    sentences = nltk.sent_tokenize(text)
                     # yield sentence and label
                    for sent in sentences:
                        if sent.strip():
                            yield {
                                "text": sent,
                                "label": label
                            }
                except (OSError, UnicodeDecodeError) as e:
                    print(f"Could not read file {item.name}: {e}")
    else:
        print(f"Error: Given directory not found: {scan_dir}")


if __name__=="__main__":
    print("Loading raw dataset ...")
    # paths to OneStopEnglish Corpus
    file_label_map = {
        0: "./datasets/OneStopEnglish/Ele-Txt",
        1: "./datasets/OneStopEnglish/Int-Txt",
        2: "./datasets/OneStopEnglish/Adv-Txt"
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

    label_names = ["Ele", "Int", "Adv"]
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=label_names)
    })
    raw_dataset = raw_dataset.cast(features)

    print("Loading raw dataset done")
    print("Saving as HF dataset...")
    raw_dataset.save_to_disk("./results/OneStopEnglish_raw")  # only contains labels and sentences