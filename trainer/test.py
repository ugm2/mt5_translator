  
import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

source = "english"
target = "spanish"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "outputs", args=model_args)

eval_df = pd.read_csv("data/eng-spa/eval.tsv", sep="\t").astype(str)

target_truth = [eval_df.loc[eval_df["prefix"] == f"translate {source} to {target}"]["target_text"].tolist()]
to_target = eval_df.loc[eval_df["prefix"] == f"translate {source} to {target}"]["input_text"].tolist()

source_truth = [eval_df.loc[eval_df["prefix"] == f"translate {target} to {source}"]["target_text"].tolist()]
to_source = eval_df.loc[eval_df["prefix"] == f"translate {target} to {source}"]["input_text"].tolist()

# Predict
target_preds = model.predict(to_target)

eng_spa_bleu = sacrebleu.corpus_bleu(target_preds, target_truth)
print("--------------------------")
print(f"{source} to {target}: ", eng_spa_bleu.score)

source_preds = model.predict(to_source)

spa_eng_bleu = sacrebleu.corpus_bleu(source_preds, source_truth)
print(f"{target} to {source}: ", spa_eng_bleu.score)