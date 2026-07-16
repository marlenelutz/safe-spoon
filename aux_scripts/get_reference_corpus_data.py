import pandas as pd
import pathlib as Pathlib
from safe_spoon.utils.common import load_yaml_config_file
from safe_spoon.utils.data_utils import (
    DEMO_REGEX,
    NAME_PATTERN,
    remove_contained_in,
    remove_empties,
    remove_exact_duplicates,
    remove_name_pattern,
    remove_near_duplicates,
)

_cfg = load_yaml_config_file()
SIMILARITY_THRESHOLD = _cfg["similarity_threshold"]


def filter_df(df: pd.DataFrame, content_col: str, label) -> pd.DataFrame:
    """Apply the same filtering pipeline used in aux_scripts/data_filtering.py."""
    print(f"df len is {len(df)} before removing empties for {label}")

    df = remove_empties(df, content_col)
    print(f"df len is {len(df)} after removing empties for {label}")

    df, _ = remove_exact_duplicates(df.reset_index(drop=True), content_col)
    print(f"df len is {len(df)} after removing exact duplicates for {label}")

    df = remove_name_pattern(df.reset_index(drop=True), content_col, NAME_PATTERN)
    print(f"df len is {len(df)} after removing NAME_X pattern for {label}")

    df = remove_name_pattern(df.reset_index(drop=True), content_col, DEMO_REGEX)
    print(f"df len is {len(df)} after removing demographic self-disclosure for {label}")

    df, _ = remove_contained_in(df.reset_index(drop=True), content_col)
    print(f"df len is {len(df)} after removing contained-in rows for {label}")

    df, _ = remove_near_duplicates(df.reset_index(drop=True), content_col, SIMILARITY_THRESHOLD)
    print(f"df len is {len(df)} after removing near duplicates for {label}")

    return df.reset_index(drop=True)


csv_files = sorted(Pathlib.Path("data/dataset/responses_without_search_tool").glob("*.csv"))

dfs = [filter_df(pd.read_csv(f), "query", f) for f in csv_files]

# check that all dfs have the same prompt_id
prompt_ids = [set(df["prompt_id"].dropna().astype(str).unique()) for df in dfs]
reference_prompt_ids = prompt_ids[0]

if not all(ids == reference_prompt_ids for ids in prompt_ids):
    print("Not all dataframes have the same prompt_id")
    # check differences in prompt_id
    for i, ids in enumerate(prompt_ids):
        if ids != reference_prompt_ids:
            missing_prompt_ids = sorted(reference_prompt_ids - ids)
            extra_prompt_ids = sorted(ids - reference_prompt_ids)
            print(f"Dataframe {i} is missing {len(missing_prompt_ids)} prompt_id values: {missing_prompt_ids[:10]}")
            print(f"Dataframe {i} has {len(extra_prompt_ids)} extra prompt_id values: {extra_prompt_ids[:10]}")

dfs = [df.sort_values("prompt_id").reset_index(drop=True) for df in dfs]

# Build one query row per prompt_id across all files, then append one response row per file.

query_df = pd.concat([df[["prompt_id", "query"]] for df in dfs], ignore_index=True)
query_df = query_df.drop_duplicates(subset=["prompt_id"], keep="first")
query_df.rename(columns={"prompt_id": "id", "query": "text"}, inplace=True)

response_dfs = []
for df, f in zip(dfs, csv_files):
    response_df = df[["prompt_id", "response"]].copy()
    response_df["id"] = response_df["prompt_id"].astype(str) + "_" + f.stem
    response_df.drop(columns=["prompt_id"], inplace=True)
    response_df.rename(columns={"response": "text"}, inplace=True)
    response_dfs.append(response_df)

final_df = pd.concat([query_df] + response_dfs, ignore_index=True)

# remove those rows (that come from the responses) in which the text starts with "I cannot fulfill this request" or only have one word
final_df = final_df[~final_df["text"].str.startswith("I cannot fulfill this request", na=False)]
final_df = final_df[final_df["text"].str.split().str.len() > 1]

# check that there are no duplicate text values across the final_df
if final_df["text"].duplicated().any():
    print("There are duplicate text values across the final dataframe")
    duplicates = final_df[final_df["text"].duplicated(keep=False)]
    print(duplicates)
    
# read additionally data/high_risk_automatically_labelled_filtered.csv and add to final_df, but only if the prompt_id is not already in final_df
high_risk_df = pd.read_csv("data/dataset/automatically-labeled-data/high_risk_automatically_labelled_filtered.csv", encoding="latin-1")

high_risk_df = filter_df(high_risk_df, "content", "high_risk_df")

final_df_with_hr = pd.concat([final_df, high_risk_df[["prompt_id", "content"]].rename(columns={"prompt_id": "id", "content": "text"})], ignore_index=True)

# remove duplicates again after adding high_risk_df
print(f"final_df_with_hr len is {len(final_df_with_hr)} before removing empties")
final_df_with_hr, _ = remove_exact_duplicates(final_df_with_hr, "text")
print(f"final_df_with_hr len is {len(final_df_with_hr)} after removing exact duplicates")

final_df_with_hr.to_csv("data/reference_corpus.csv", index=False)