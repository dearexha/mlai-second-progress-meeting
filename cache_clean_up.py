import datasets
from datasets import load_from_disk

# This function scans the cache and removes all unreferenced files.
# It returns the number of files deleted and the total space freed.
dataset_names = ["SimpleWikipedia_raw", "OneStopEnglish_raw", "SimpleWikipedia_tokenized_and_measured", "OneStopEnglish_tokenized_and_measured", "SimpleWikipedia_filtered", "OneStopEnglish_filtered"]

print("Cleaning up dataset caches ...")

for ds_name in dataset_names:
    ds = load_from_disk(f"./results/hf_datasets/{ds_name}")
    ds.cleanup_cache_files()

print("Cleaning up dataset caches done")






