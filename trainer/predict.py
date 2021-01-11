import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

source = "english"
target = "spanish"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 512
model_args.min_length = 0
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "outputs", args=model_args)


eval_df = pd.read_csv("data/eng-spa/eval.tsv", sep="\t").astype(str)

target_truth = [eval_df.loc[eval_df["prefix"] == f"translate {source} to {target}"]["target_text"].tolist()]
to_target = eval_df.loc[eval_df["prefix"] == f"translate {source} to {target}"]["input_text"].tolist()

# print(to_target[:2])
sentences = [
    "Me gusta tocar muchos instrumentos. Adoro la música",
    "I like to play many instruments. I love music"
]
target_preds = model.predict(sentences)
print(target_preds)