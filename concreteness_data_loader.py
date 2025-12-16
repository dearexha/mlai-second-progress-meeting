import pandas as pd

def load_concreteness_lexicon(filepath="concreteness_words.csv"):
    """
    Load concreteness lexicon. Expected columns: Word, Conc.M
    Returns: dict {word_lowercase: conc_value}
    """
    df = pd.read_csv(filepath)
    conc_dict = {}
    for _, row in df.iterrows():
        word = str(row['Word']).lower().strip()
        if 'Conc.M' in df.columns:
            conc_dict[word] = float(row['Conc.M'])
    return conc_dict