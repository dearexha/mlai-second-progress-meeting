import pandas as pd

def load_aoa_lexicon(filepath="aoa_words.csv"):
    """
    Load AoA lexicon from aoa_words.csv.
    
    Expected columns: Word, AoA_Kup (Kuperman et al. Age of Acquisition ratings)
    The file also contains other AoA columns (AoA_Kup_lem, AoA_Bird_lem, etc.) but
    we use AoA_Kup as it matches the paper reference (Kuperman et al., 2012).
    
    Returns: dict {word_lowercase: aoa_value}
    Only includes words that have non-null AoA_Kup values.
    """
    df = pd.read_csv(filepath)
    aoa_dict = {}
    for _, row in df.iterrows():
        word = str(row['Word']).lower().strip()
        # Use AoA_Kup column (Kuperman et al. ratings)
        # Skip rows where AoA_Kup is NaN
        if 'AoA_Kup' in df.columns and pd.notna(row['AoA_Kup']):
            aoa_dict[word] = float(row['AoA_Kup'])
    return aoa_dict
