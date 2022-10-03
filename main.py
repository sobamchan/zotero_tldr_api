from argparse import ArgumentParser

from pyzotero import zotero
from schnitsum import SchnitSum

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="sobamchan/bart-large-scitldr"
    )
    parser.add_argument("--user-id", type=str, required=True)
    parser.add_argument("--zotero-key", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    model = SchnitSum(args.model_name, use_gpu=False)

    zot = zotero.Zotero(args.user_id, "user", args.zotero_key)

    items = zot.top(limit=args.limit, itemType="preprint || conferencePaper || journalArticle")

    for item in items:
        notes = zot.children(item["key"], itemType="Note", tag="tldr")
        if notes:
            continue
        else:
            newnote = zot.item_template("note")
            newnote["tags"].append({"tag": "tldr"})
            if "abstractNote" in item["data"]:
                abst = item["data"]["abstractNote"]
                newnote["note"] = model([abst])[0]
            zot.create_items([newnote], item["key"])
