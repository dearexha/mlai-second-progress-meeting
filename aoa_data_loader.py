def load_aoa_lexicon(filepath="AoA_ratings_Kuperman_et_al_BRM.csv"):
    """
    Load AoA lexicon. Expected columns: Word, Rating.Mean
    Returns: dict {word_lowercase: aoa_value}
    """
    df = pd.read_csv(filepath)
    aoa_dict = {}
    for _, row in df.iterrows():
        word = str(row['Word']).lower().strip()
        # Some datasets use 'Rating.Mean', some use 'AoA'
        if 'Rating.Mean' in df.columns:
            aoa_dict[word] = float(row['Rating.Mean'])
        elif 'AoA' in df.columns:
            aoa_dict[word] = float(row['AoA'])
    return aoa_dict
