"""
Implements LMPriors from https://arxiv.org/abs/2210.12530
"""
import os
import sys
import warnings
import numpy as np
import logging
from langchain.prompts import PromptTemplate
from openai import OpenAI
from tqdm import tqdm
import json
import pickle as pkl
from constants import OPENAI_API
warnings.filterwarnings("ignore") # suppress warnings


def create_lmpriors_prompt(prompt_filename, feature, description, category = "", display = False):
    """
    Create Lmpriors prompt.
        - prompt_filename should be a .txt file.
        - feature (str): feature name
        - description (str): descripition of the feature
        - category (str): string description of the classification category in question
        - display (bool): display the prompt or not
    """
    with open(prompt_filename, "r", encoding="utf-8") as file:
        lmpriors_prompt_template = file.read()
    
    # Create the PromptTemplate object
    lmpriors_prompt = PromptTemplate(
        input_variables=["feature", "description", "category"],
        template=lmpriors_prompt_template
    )

    filled_prompt = lmpriors_prompt.format(
            feature = feature,
            description= description,
            category = category
        )
    
    if display:
        print(filled_prompt)
    
    return filled_prompt


def lmpriors_criterion(token, token_logprob):
    token_prob = np.exp(token_logprob)
    # Clip to avoid log(0)
    token_prob = np.clip(token_prob, 1e-12, 1 - 1e-12)
    # Compute log-odds
    logit = token_logprob - np.log(1 - token_prob)
    return logit if token == 'T' else -logit


# Wipe all files in save_dir related to this method
def wipe_save_dir(save_dir):
    files_to_remove = [
        "selected_features_lmpriors.pkl",
        "selected_features_lmpriors.txt",
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")


def query_lmpriors(
    prompt_filename: str,
    feature_description_filename: str, # this should be the path to a json file containing feature descriptions
    save_dir: str,
    model = "gpt-4o", # str, an openai model that offers log probability output
    category = "",
    threshold = 0.2, # threshold to include a feature or not # tunable, 0.2 roughly corresponds to 60% T and 40% F.
    display = False, # display output token and probability or now
    wipe = False,
    max_retries=5  # maximum retry attempts
):
    """
    Get LMpriors-selected features, ranked by criterion score, with progress saved incrementally.
    Only save final sorted .txt file at the end.
    """
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "selected_features.pkl")
    txt_path = os.path.join(save_dir, "selected_features.txt")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API))

    # Load previous progress if it exists
    if not wipe and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            selected_feature_tuples = pkl.load(f)
        processed_feats = {feat for feat, _ in selected_feature_tuples}
        print(f"Resuming with {len(selected_feature_tuples)} features already selected.")
    else:
        selected_feature_tuples = []
        processed_feats = set()
        if wipe:
            print("🧹 Wiping previous results.")
        with open(pkl_path, "wb") as f:
            pkl.dump([], f)
        if os.path.exists(txt_path):
            os.remove(txt_path)

    with open(feature_description_filename, "r") as f:
        feat_desc = json.load(f)

    for feat, desc in tqdm(feat_desc.items(), desc='Processing features...'):
        if feat in processed_feats:
            continue  # Skip already processed

        retries = 0
        valid_token = False

        while retries < max_retries and not valid_token:
            prompt = create_lmpriors_prompt(prompt_filename, feat, desc, category)
            system_message = "Complete the user prompt, output only 'T' or 'F'."
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1,
                    logprobs=True,
                    temperature=0
                )

                token_info = response.choices[0].logprobs.content[0]
                generated_token = token_info.token.strip()
                token_logprob = token_info.logprob

                if display:
                    print(f"\nGenerated token: {generated_token}")
                    print(f"Log probability: {token_logprob}")

                if generated_token in {"T", "F"}:
                    valid_token = True
                else:
                    retries += 1
                    print(f"⚠️ Invalid token for '{feat}' → '{generated_token}'. Retrying ({retries}/{max_retries})...")

            except Exception as e:
                retries += 1
                print(f"API error on '{feat}': {e}. Retrying ({retries}/{max_retries})...")

        if not valid_token:
            print(f"Failed on feature '{feat}' after {max_retries} retries.")
            sys.exit(-1)

        criterion = lmpriors_criterion(generated_token, token_logprob)
        if criterion >= threshold:
            selected_feature_tuples.append((feat, criterion))

            # Save progress incrementally to .pkl
            with open(pkl_path, "wb") as f:
                pkl.dump(selected_feature_tuples, f)

    # Final sort and output
    selected_feature_tuples.sort(key=lambda x: -x[1])  # descending by criterion
    selected_feature_ls = [feat for feat, _ in selected_feature_tuples]

    # Save ranked feature names to .txt (one-time final write)
    with open(txt_path, "w") as f:
        for feat in selected_feature_ls:
            f.write(feat + "\n")

    return selected_feature_ls