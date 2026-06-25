import pandas as pd
import pathlib as Pathlib
from safe_spoon.utils.data_utils import remove_empties, remove_exact_duplicates, remove_near_duplicates

csv_files = sorted(Pathlib.Path("data/responses_with_search_tool").glob("*.csv"))

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    print(f"df len is {len(df)} before removing empties for {f}")
    
    df = remove_empties(df, "query")
    print(f"df len is {len(df)} after removing empties for {f}")
   
    df, _ = remove_exact_duplicates(df, "query")
    print(f"df len is {len(df)} after removing exact duplicates for {f}")
   
    df, _ = remove_near_duplicates(df, "query", 0.95)
    print(f"df len is {len(df)} after removing near duplicates for {f}")
    import pdb; pdb.set_trace()
    dfs.append(df)

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
high_risk_df = pd.read_csv("data/high_risk_automatically_labelled_filtered.csv", encoding="latin-1")

print(f"high_risk_df len is {len(high_risk_df)} before removing empties")

high_risk_df = remove_empties(high_risk_df, "content")
print(f"high_risk_df len is {len(high_risk_df)} after removing empties")

high_risk_df, _ = remove_exact_duplicates(high_risk_df, "content")
print(f"high_risk_df len is {len(high_risk_df)} after removing exact duplicates")

high_risk_df, _ = remove_near_duplicates(high_risk_df, "content", 0.95)
print(f"high_risk_df len is {len(high_risk_df)} after removing near duplicates")

final_df_with_hr = pd.concat([final_df, high_risk_df[["prompt_id", "content"]].rename(columns={"prompt_id": "id", "content": "text"})], ignore_index=True)

# remove duplicates again after adding high_risk_df
print(f"final_df_with_hr len is {len(final_df_with_hr)} before removing empties")
final_df_with_hr, _ = remove_exact_duplicates(final_df_with_hr, "text")
print(f"final_df_with_hr len is {len(final_df_with_hr)} after removing exact duplicates")

final_df_with_hr.to_csv("data/reference_corpus.csv", index=False)