
import nltk
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
import numpy as np
import pyphen
from tqdm import tqdm
import torch
import math
from transformers import BertTokenizerFast, BertForMaskedLM
from torch.nn import CrossEntropyLoss
from aoa_data_loader import load_aoa_lexicon
from concreteness_data_loader import load_concreteness_lexicon


# nltk.download('punkt')
# nltk.download('punkt_tab')



# can take batched input/ multiple sentences
def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

def compute_sentence_length(sent_word_list):
    return len(sent_word_list)


def compute_word_rarity(sent_word_list, corpus_vocab_counts):
    if len(sent_word_list) == 0: # in case the sentence contains no words
        return np.nan
    counts = np.array([corpus_vocab_counts[word] for word in sent_word_list])
    N_total = np.sum(list(corpus_vocab_counts.values()))
    # N_total = len(corpus_vocab_counts.keys())
    return - 1/len(sent_word_list) * np.sum(np.log(counts/N_total))


def compute_fre_score(sent_word_list, dic):
    if len(sent_word_list) == 0:
        return np.nan
    syllable_count = []
    for word in sent_word_list:
        syllable_list = dic.inserted(word).split("-")
        syllable_count.append(len(syllable_list))
    avg_syllables_per_word = np.mean(syllable_count)
    sentence_length = len(sent_word_list)
    return 206.835 - 1.015 * sentence_length - 84.6 * avg_syllables_per_word

def compute_Shannon_entropy(sent_word_list):
    if len(sent_word_list) == 0:
        return np.nan
    sent_length = len(sent_word_list)
    sentence_hist = Counter(sent_word_list)
    p = np.array(list(sentence_hist.values()))/len(sent_word_list)
    return -np.sum(p * np.log2(p))

def compute_TTR(sent_word_list): # #unique_words/#total_words
    if len(sent_word_list) == 0:
        return np.nan
    return len(np.unique(sent_word_list))/len(sent_word_list)


def apply_metric_to_dataset(batch, metric_name, metric_func, metric_func_kwargs, words_are_tokens, tokenizer): # applies a given difficulty measure to the HF dataset, compatible with .map()
    metric_values = []  # values of the metric for the current batch of sentences

    if words_are_tokens:
        for id_list in batch["input_ids"]:
            token_list = tokenizer.convert_ids_to_tokens(id_list)
            metric_values.append(metric_func(token_list, **metric_func_kwargs))
        col_name = metric_name + "_tokens"
    else:
        for sent in batch["text"]:
            word_list = nltk.word_tokenize(sent)
            word_list = [w for w in word_list if w.isalpha()]  # comment out if words with non alphabetic letters are also of interest
            metric_values.append(metric_func(word_list, **metric_func_kwargs))
        col_name = metric_name + "_words"
    
    return {col_name: metric_values} # adds new column to the HF dataset


# we use a separate method for language model score computation, since it can only be computed for bert-tokens
def compute_perplexity(example, model, device, tokenizer):
    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
    
    # sequence length (includes [CLS] and [SEP])
    seq_len = input_ids.shape[1]
    
    # return None as a value for sentence that only contain CLS and SEP
    if seq_len < 3: 
        return {"perplexity": None}

    # to compute perplexity we create a batch of the same sentence, where in each instance a different token is masked (excluding CLS and SEP)
    n_tokens_to_mask = seq_len - 2
    repeat_input_ids = input_ids.repeat(n_tokens_to_mask, 1)
    mask_indices = torch.arange(1, seq_len - 1).to(device)
    repeat_input_ids[torch.arange(n_tokens_to_mask), mask_indices] = tokenizer.mask_token_id
    
    # create the labels for computation of the NLL/cross entropy (Pytorch convention: ignore_index=-100)
    labels = torch.full(repeat_input_ids.shape, -100).to(device)
    labels[torch.arange(n_tokens_to_mask), mask_indices] = input_ids[0, mask_indices]

    # instead of processing the entire n_tokens many sentences as a single batch, process in smaller batches to avoid memory problems on gpu
    inference_batch_size = 32
    total_nll = 0.0
    with torch.no_grad():
        for i in range(0, n_tokens_to_mask, inference_batch_size):
            # note: slices beyond the maximum index are ignored
            batch_input = repeat_input_ids[i : i + inference_batch_size]
            batch_labels = labels[i : i + inference_batch_size]
            
            # forward pass through the bert model
            outputs = model(batch_input, labels=batch_labels)
            
            # outputs.loss is avg nll, we multiply with the batchsize to obtain the sum
            total_nll += outputs.loss.item() * batch_input.size(0)

    # PPL = exp(avg_NLL)
    avg_nll = total_nll / n_tokens_to_mask  # normalize sum over NLLs
    ppl = math.exp(avg_nll)
    
    return {"perplexity": ppl, "log_perplexity": avg_nll}

def compute_aoa_score(sent_word_list, aoa_lexicon):
    """
    Compute average Age of Acquisition for a sentence.
    
    CRITICAL ANALYSIS:
    - Simple averaging may not be ideal: common function words (the, a, is) 
      learned early (low AoA) but don't contribute much to difficulty
    - Default value of 25.0 is reasonable (most words learned before 25)
    - Should consider: only content words? Weight by frequency?
    
    IMPROVEMENTS MADE:
    - Only compute AoA for words actually in lexicon (exclude missing words from average)
    - This gives more accurate scores when lexicon coverage is good
    - Returns NaN if no words found in lexicon (better than using all defaults)
    
    Current implementation: Mean of words found in lexicon only.
    """
    if len(sent_word_list) == 0:
        return np.nan

    aoa_values = []
    
    for word in sent_word_list:
        # Use lowercase to match lexicon
        word_lower = word.lower()
        if word_lower in aoa_lexicon:
            aoa_values.append(aoa_lexicon[word_lower])
    
    # Only return mean if we found at least some words in lexicon
    # Otherwise return NaN (better than using all defaults)
    if len(aoa_values) == 0:
        return np.nan
    
    # Return mean AoA (lower = easier, higher = harder)
    return np.mean(aoa_values)

def compute_concreteness_score(sent_word_list, conc_lexicon):
    """
    Compute average concreteness for a sentence.
    
    CRITICAL REVIEW NOTES:
    - Averaging concreteness is standard practice
    - Default value of 3.0 is neutral (midpoint of 1-5 scale)
      * This is reasonable for unknown words
    - Lower concreteness (more abstract) may correlate with difficulty
      * But this depends on context and task
    - Consider: abstract words in simple contexts might still be easy
    
    Args:
        sent_word_list: List of words (already filtered to alphabetic only)
        conc_lexicon: Dict mapping lowercase words to concreteness values (1-5 scale)
    
    Returns:
        Average concreteness score (higher = more concrete = potentially easier)
    """
    if len(sent_word_list) == 0:
        return np.nan

    conc_values = []
    for word in sent_word_list:
        word_lower = word.lower()
        # Default 3.0 is neutral (midpoint of 1-5 scale)
        conc_values.append(conc_lexicon.get(word_lower, 3.0))

    return np.mean(conc_values)

if __name__ == "__main__":

    dataset_name = "SimpleWikipedia"
    # dataset_name = "OneStopEnglish"
    num_proc = 8  # number of processors to use for computing the metrics/ tokenization

    print("Loading raw dataset ...")
    raw_dataset = load_from_disk(f"./results/hf_datasets/{dataset_name}_raw")
    print("Loading raw dataset done")

    print("Tokenizing dataset ...")
    # currently the dataset only contains the sentence and its respective label
    # now we add a new column that contains the BERT tokens for each sentence
    model_id = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_id)

    print("Loading AoA and Concreteness lexicons...")
    aoa_lexicon = load_aoa_lexicon("./lexicons/AoA_ratings_Kuperman_et_al_BRM.csv")
    conc_lexicon = load_concreteness_lexicon("./lexicons/Concreteness_ratings_Brysbaert_et_al_BRM.csv")
    print(f"AoA lexicon size: {len(aoa_lexicon)}")
    print(f"Concreteness lexicon size: {len(conc_lexicon)}")
    

    data_set = raw_dataset.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer}
    )

    print("Tokenizing dataset done")

    print("Computing vocabulary histograms ...")

    # compute vocabulary counts for words and tokens over the entire corpus
    label_feature = raw_dataset.features["label"]
    label_names = label_feature.names

    word_corpus_vocab_hist_list = [Counter() for label in label_names]
    token_corpus_vocab_hist_list = [Counter() for label in label_names]

    for example in tqdm(data_set):
        # compute the word histogram
        word_list = nltk.word_tokenize(example["text"])
        word_list = [word for word in word_list if word.isalpha()] # comment out if words with non alphabetic letters are also of interest
        word_hist = Counter(word_list)

        # compute the token histogram
        token_hist = Counter(tokenizer.convert_ids_to_tokens(example["input_ids"]))

        # add to vocab histogram for the specific class
        word_corpus_vocab_hist_list[example["label"]].update(word_hist)
        token_corpus_vocab_hist_list[example["label"]].update(token_hist)

    
    word_corpus_vocab_counts = sum(word_corpus_vocab_hist_list, Counter())
    token_corpus_vocab_counts = sum(token_corpus_vocab_hist_list, Counter())

    print("Computing vocabulary histograms done")

    print("Computing traditional evaluation metrics ...")

    dic = pyphen.Pyphen(lang="en_US")

    metric_and_kwargs_list = [{"metric_name":"sentence_length", "metric_func":compute_sentence_length, "metric_func_kwargs":{}},
                              {"metric_name":"word_rarity", "metric_func":compute_word_rarity, "metric_func_kwargs":{"corpus_vocab_counts": word_corpus_vocab_counts}},
                              {"metric_name":"fre_score", "metric_func":compute_fre_score, "metric_func_kwargs":{"dic": dic}},
                              {"metric_name":"shannon_entropy", "metric_func":compute_Shannon_entropy, "metric_func_kwargs":{}},
                              {"metric_name":"ttr", "metric_func":compute_TTR, "metric_func_kwargs":{}},
                              {"metric_name": "aoa_score", "metric_func": compute_aoa_score, "metric_func_kwargs": {"aoa_lexicon": aoa_lexicon}},
                              {"metric_name": "concreteness_score", "metric_func": compute_concreteness_score, "metric_func_kwargs": {"conc_lexicon": conc_lexicon}}
                              ]
    
    # compute the metrics based on words
    for metric in metric_and_kwargs_list:
        data_set = data_set.map(
            apply_metric_to_dataset,
            batched=True,
            num_proc=num_proc,
            fn_kwargs={**metric, "words_are_tokens":False, "tokenizer": tokenizer}
        )

    # compute the metrics based on tokens
    for metric in metric_and_kwargs_list:
        if metric["metric_name"] == "word_rarity":
            data_set = data_set.map(
                apply_metric_to_dataset,
                batched=True,
                num_proc=num_proc,
                fn_kwargs={"metric_name":"word_rarity", "metric_func":compute_word_rarity, "metric_func_kwargs":{"corpus_vocab_counts": token_corpus_vocab_counts}, "words_are_tokens":True, "tokenizer": tokenizer}
            )
        else:
            data_set = data_set.map(
                apply_metric_to_dataset,
                batched=True,
                num_proc=num_proc,
                fn_kwargs={**metric, "words_are_tokens":True, "tokenizer": tokenizer}
            )
    print("Computing traditional evaluation metrics done")

    # compute average perplexity for each sentence from a pretrained bert model (language model score)
    print("Computing Language Model Score ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForMaskedLM.from_pretrained(model_id).to(device)
    model.eval()
    data_set = data_set.map(compute_perplexity, batched=False, fn_kwargs={"model": model, "device": device, "tokenizer": tokenizer})

    print("Computing Language Model Score done")

    print("Saving results to disk ...")

    # save HF dataset to disk
    data_set.save_to_disk(f"./results/{dataset_name}_tokenized_and_measured")

    print("Saving results to disk done")


