import shutil
import os
from datasets import load_from_disk
import gc
import numpy as np
from collections import Counter
from matplotlib.axes import Axes
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from tqdm import tqdm

def update_existing_dataset(dataset, path):
    tmp_path = "./results/tmp/hf_tmp"
    dataset.save_to_disk(tmp_path)

    del dataset
    gc.collect()

    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp_path, path)

def compute_histograms(dataset_name, excluded_columns=["text", "label", "input_ids", "token_type_ids", "attention_mask"], save_to_disk=True, bounds={}):
    dataset = load_from_disk(f"./results/hf_datasets/{dataset_name}")

    print("Computing histograms ...")
    label_feature = dataset.features["label"]
    label_names = label_feature.names
    column_names = [col for col in dataset.features.keys() if col not in excluded_columns]
    metric_hists = {
        col:[Counter() for label in label_names] for col in column_names
    } 
    for example in tqdm(dataset):
        # ignore sentences of length 0 (if sentence_length_words exists)
        # Otherwise, skip empty text
        if "sentence_length_words" in example:
            if example["sentence_length_words"] == 0:
                continue
        elif "text" in example and len(example["text"].strip()) == 0:
            continue
        
        for col in column_names:
            # Skip NaN values
            if col not in example or (isinstance(example[col], float) and (np.isnan(example[col]) or np.isinf(example[col]))):
                continue
            if col in bounds:
                if bounds[col]["min"] <= example[col] <= bounds[col]["max"]:
                    metric_hists[col][example["label"]][example[col]] += 1
            else:
                metric_hists[col][example["label"]][example[col]] += 1
    print("Computing histograms done")

    if save_to_disk:
        print("Saving results to disk ...")
        eval_dict = dict()
        eval_dict["metric_hists"] = metric_hists
        with open(f"./results/{dataset_name}{"_Filtered" if len(bounds)!=0 else ""}_eval_dict.json", "w") as f:
            json.dump(eval_dict, f, indent=4)
        print("Saving results to disk done")
    
    return eval_dict



def plot_agg_histogram(ax : Axes, hist_counter_list : list[Counter], xlabel, color_list, dataset_labels, num_bins=40):

    quantity_list = [np.array(list(hist_counter.keys()), dtype=float) for hist_counter in hist_counter_list]
    counts_list = [np.array(list(hist_counter.values()), dtype=float) for hist_counter in hist_counter_list]

    min_val = min(np.min(arr) for arr in quantity_list)
    max_val = max(np.max(arr) for arr in quantity_list)

    bins = np.linspace(min_val, max_val, num_bins+1)

    binned_counts_list = [np.histogram(quantity_list[i], bins=bins, weights=counts_list[i])[0] for i in range(len(hist_counter_list))]
    bin_edges = bins[:-1]  # take the left bind edge for plotting

    for i in range(len(binned_counts_list)):
        ax.fill_between(bin_edges, binned_counts_list[i], color=color_list[i], alpha=0.3, label=dataset_labels[i])
        ax.plot(bin_edges, binned_counts_list[i], color=color_list[i], alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True)
    # ax.set_title(xlabel)
    # ax.legend()
    return

def plot_metrics_hist(dataset_name, selected_metrics, cols=3, base_size=(4, 3), share_y=True, filtered=False):
    dataset = load_from_disk(f"./results/hf_datasets/{dataset_name}")
    label_feature = dataset.features["label"]
    label_names = label_feature.names

    with open(f"./results/{dataset_name}{"_Filtered" if filtered else ""}_eval_dict.json", "r") as f:
        eval_dict = json.load(f)

    num_plots = len(selected_metrics)
    rows = math.ceil(num_plots / cols)
    figsize = (cols * base_size[0], rows * base_size[1])
    fig, axs = plt.subplots(nrows=rows, ncols=cols, sharey=share_y, figsize=figsize, squeeze=False, layout="constrained")
    formatter = ticker.EngFormatter()

    axs_flat = axs.flatten()
    for i in range(len(selected_metrics)):
        plot_agg_histogram(axs_flat[i], eval_dict["metric_hists"][selected_metrics[i]], selected_metrics[i], ["blue", "green", "purple"], label_names, num_bins=40)
    axs_flat[0].yaxis.set_major_formatter(formatter)
    axs_flat[0].legend()

    for i in range(num_plots, len(axs_flat)):
        axs_flat[i].axis('off')

    plt.show()
