import json
from argparse import ArgumentParser
from typing import List

import requests
import torch
from transformers import (AutoTokenizer, BartConfig,
                          BartForConditionalGeneration)


class Model:
    def __init__(self, model_path):
        print("initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        bart = BartForConditionalGeneration(BartConfig())
        bart.load_state_dict(
            torch.load(model_path),
            strict=False,
        )
        self.bart = bart
        print("loaded!")

    def summarize(self, text: str):

        inputs = self.tokenizer(
            [text], padding="max_length", truncation=True, return_tensors="pt"
        )
        summary_ids = self.bart.generate(
            inputs["input_ids"],
            max_length=50,
            num_beams=1,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]


def get_items(user_id: int, zotero_key: str):
    url = f"https://api.zotero.org/users/{user_id}/items?sort=dateAdded&direction=desc&itemType=conferencePaper || journalArticle&limit=50"
    headers = {"Authorization": f"Bearer {zotero_key}"}
    response = requests.request("GET", url, headers=headers, data={})
    data = response.json()
    return data


def create_notes(items: List, user_id: int, zotero_key: str):
    url = f"https://api.zotero.org/users/{user_id}/items"

    payload = [
        {
            "itemType": "note",
            "parentItem": item["key"],
            "note": item["tldr"],
            "tags": [],
            "collections": [],
            "relations": {},
        }
        for item in items
    ]

    headers = {
        "Authorization": f"Bearer {zotero_key}",
        "Content-Type": "application/json",
    }

    requests.request("POST", url, headers=headers, data=json.dumps(payload))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--user-id", type=str)
    parser.add_argument("--zotero-key", type=str)
    args = parser.parse_args()

    model = Model(args.model_path)
    items = get_items(args.user_id, args.zotero_key)

    with open("key_list.txt") as f:
        exclude_keys = [l.strip() for l in f.readlines()]

    keys = []
    new_items = []
    for item in items:
        data = item["data"]
        abst = data["abstractNote"]
        key = data["key"]

        if key in exclude_keys:
            continue

        if len(abst.split()) <= 3:
            keys.append(key)
            continue

        tldr = model.summarize(abst)
        new_items.append({"key": key, "tldr": tldr})
        keys.append(key)

    create_notes(new_items, args.user_id, args.zotero_key)

    with open("./key_list.txt", "a") as f:
        f.write("\n")
        f.write("\n".join(keys))
