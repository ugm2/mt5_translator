import os
import pandas as pd

def load_file(data_path, filename, max_lines=10000000000):
    text_lines = []
    with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            text_lines += [line.strip("\n")]
            if i+1==max_lines: break
    return text_lines

def prepare_translation_datasets(data_path, source, target, max_train_lines=1000000):
    target_text = load_file(data_path, "train.trg", max_train_lines)

    source_text = load_file(data_path, "train.src", max_train_lines)

    data = []
    for t, s in zip(target_text, source_text):
        data.append([f"translate {target} to {source}", t, s])
        data.append([f"translate {source} to {target}", s, t])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
    print("Training length: ", len(train_df))

    target_text = load_file(data_path, "test.trg")

    source_text = load_file(data_path, "test.src")

    data = []
    for t, s in zip(target_text, source_text):
        data.append([f"translate {target} to {source}", t, s])
        data.append([f"translate {source} to {target}", s, t])

    eval_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
    print("Test length: ", len(eval_df))

    return train_df, eval_df

train_df, eval_df = prepare_translation_datasets("data/eng-spa",
                                                 source="english",
                                                 target="spanish",
                                                 max_train_lines=10000)

train_df.to_csv("data/eng-spa/train.tsv", sep="\t")
eval_df.to_csv("data/eng-spa/eval.tsv", sep="\t")