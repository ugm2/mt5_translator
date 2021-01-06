import os
import pandas as pd

def prepare_translation_datasets(data_path):
    with open(os.path.join(data_path, "train.trg"), "r", encoding="utf-8") as f:
        sinhala_text = f.readlines()
        sinhala_text = [text.strip("\n") for text in sinhala_text]

    with open(os.path.join(data_path, "train.src"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for sinhala, english in zip(sinhala_text, english_text):
        data.append(["translate sinhala to english", sinhala, english])
        data.append(["translate english to sinhala", english, sinhala])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    with open(os.path.join(data_path, "test.trg"), "r", encoding="utf-8") as f:
        sinhala_text = f.readlines()
        sinhala_text = [text.strip("\n") for text in sinhala_text]

    with open(os.path.join(data_path, "test.src"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for sinhala, english in zip(sinhala_text, english_text):
        data.append(["translate sinhala to english", sinhala, english])
        data.append(["translate english to sinhala", english, sinhala])

    eval_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    return train_df, eval_df

train_df, eval_df = prepare_translation_datasets("data/eng-spa")

# train_df.to_csv("data/eng-spa/train.tsv", sep="\t")
# eval_df.to_csv("data/eng-spa/eval.tsv", sep="\t")