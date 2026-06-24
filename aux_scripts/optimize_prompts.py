"""
Script to automate the optimiztion of the prompts for topic labelling and annotation unit labelling.
"""

import os

from dotenv import load_dotenv

from safe_spoon.clustering import AnnotationUnitModel
from safe_spoon.utils.data_utils import corpus_for_category, load_corpus_df
from safe_spoon.prompting import _default_prompt_path
from safe_spoon.topic_modeling.lda import LDATopicModel
from safe_spoon.utils.common import load_annotation_unit_config

load_dotenv()

# data config
CAT = "Health"
INPUT_FILE = "data/high_risk_automatically_labelled_filtered_cleaned.csv"
#MODEL_DIR = f"./data/models/{CAT.replace(' ', '_')}"
MODEL_DIR = "data/models/Health_optimize/model_10_topics"

# llm config
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-5.4-nano-2026-03-17"
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

# paths to the prompt templates to compare
TOPIC_PROMPT_VARIANTS = {
    "default": _default_prompt_path("labelling_dft.txt"),
    # @marlenelutz add others
}

# ANN_UNIT_PROMPT_VARIANTS = {
#     "default": _default_prompt_path("unit_labelling_dft.txt"),
#     # @marlenelutz add others
# }

# # Annotation unit config
# _au_cfg = load_annotation_unit_config()
# MIN_SIZE = _au_cfg["min_size"]
# MAX_PURITY = _au_cfg["max_purity"]
# PW_MIXTURE = _au_cfg["pw_mixture"]
# PW_SIZE = _au_cfg["pw_size"]
# PW_BALANCE = _au_cfg["pw_balance"]

# Reconstruct corpus
_, _all_queries, _all_labels = load_corpus_df(INPUT_FILE)
df_corpus = corpus_for_category(_all_queries, _all_labels, CAT)

##################################################
# optimize topic labels

# load models
lda = LDATopicModel.load(MODEL_DIR, corpus=df_corpus)
lda.llm_provider   = LLM_PROVIDER
lda.llm_model_type = LLM_MODEL
lda.llm_api_key    = LLM_API_KEY
tm  = lda.tm

print("\n" + "=" * 70)
print("TOPIC LABEL VARIANTS")
print("=" * 70)

topic_results: dict = {}
for name, prompt_path in TOPIC_PROMPT_VARIANTS.items():
    print(f"\n- Variant: {name}  ({prompt_path})")
    results = tm.generate_topic_outputs(task="label", topn=10, prompt_path=prompt_path)
    topic_results[name] = results
    for tpc_id, label in results:
        print(f"  Topic {tpc_id:2d}: {label}")

##################################################
# optimize annotation unit labels
# @marlenelutz: i'm still working on this but the logic for optimization would be similar to that of the topic labels

# aum = AnnotationUnitModel.from_lda(
#     lda,
#     min_size = MIN_SIZE,
#     max_purity = MAX_PURITY,
#     pw_mixture = PW_MIXTURE,
#     pw_size = PW_SIZE,
#     pw_balance = PW_BALANCE,
#     llm_provider = LLM_PROVIDER,
#     llm_model_type = LLM_MODEL,
#     llm_api_key = LLM_API_KEY,
# )
# aum.build()

# print("\n" + "=" * 70)
# print("ANNOTATION UNIT LABEL VARIANTS")
# print("=" * 70)

# unit_results: dict = {}
# for name, prompt_path in ANN_UNIT_PROMPT_VARIANTS.items():
#     print(f"\n- Variant: {name}  ({prompt_path})")
#     results = aum.generate_unit_outputs(prompt_path=prompt_path)
#     unit_results[name] = dict(results)
#     for node_id, label in results:
#         print(f"  {node_id[:40]:<40} {label}")
