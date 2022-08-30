import datetime
import re
from os.path import dirname
from typing import Any, cast

import yaml

ROOT = dirname(dirname(__file__))

sections: dict[str, dict[str, Any]] = {
    "publications": {"title": "## 📝 Publications"},
    "applications": {"title": "## 🛠️ Applications"},
    "videos": {"title": "## 📺 Videos"},
    "packages": {"title": "## 📦 Packages"},
    "code": {"title": "## 🧑‍💻 Code"},
    "posts": {"title": "## 🌐 Blog Posts"},
}

for key in sections:
    with open(f"{ROOT}/data/{key}.yml") as file:
        sections[key]["items"] = yaml.safe_load(file.read())


seen_ids: set[str] = set()
required_keys = {"id", "title", "url", "date", "authors", "description"}
optional_keys = {"org", "authors_url", "lang", "repo", "date_added"}
valid_languages = ("PyTorch", "TensorFlow", "JAX", "Julia", "Others")


def validate_item(itm: dict[str, str]) -> None:
    """Checks that an item conforms to schema. Raises ValueError if not."""
    # no need to check for duplicate keys, YAML enforces that
    itm_keys = set(itm.keys())
    err = None

    if (id := itm["id"]) in seen_ids:
        err = f"Duplicate {id = }"
    else:
        seen_ids.add(id)

    if not id.startswith(("pub-", "app-", "vid-", "pkg-", "code-", "post-")):
        err = f"Invalid {id = }"

    if id.startswith(("pkg-", "code-")) and itm["lang"] not in valid_languages:
        err = f"Invalid lang in {id}: {itm['lang']}, must be one of {valid_languages}"

    if missing_keys := required_keys - itm_keys:
        err = f"Missing key(s) in {id}: {missing_keys}"

    if bad_keys := itm_keys - required_keys - optional_keys:
        err = f"Unexpected key(s) in {id}: {bad_keys}"

    authors = itm["authors"]
    if "et al" in authors or "et. al" in authors:
        err = (
            f"Incomplete authors in {id}: don't use 'et al' in {authors = }, list "
            "them all"
        )

    if not isinstance(itm["date"], datetime.date):
        err = f"Invalid date in {id}: {itm['date']}"

    if err:
        raise ValueError(err)


for key, sec in sections.items():
    sec["markdown"] = ""

    # keep inside outer sections loop to refill language subsections for sections
    # Code and Packages
    lang_names = ["PyTorch", "TensorFlow", "JAX", "Julia", "Others"]

    # sort first by language with order determined by lang_names (only applies to
    # Package and Code sections), then by date
    sec["items"].sort(key=lambda x: x["date"], reverse=True)
    if key in ("packages", "code"):
        sec["items"].sort(key=lambda itm: lang_names.index(itm["lang"]))  # noqa: B023

    for itm in sec["items"]:
        itm = cast(dict[str, str], itm)

        if (lang := itm.get("lang", None)) in lang_names:
            lang_names.remove(lang)
            # print subsection titles
            sec["markdown"] += (
                f'<br>\n\n### <img src="assets/{lang.lower()}.svg" alt="{lang}" '
                f'height="20px"> &nbsp;{lang} {key.title()}\n\n'
            )

        validate_item(itm)

        authors, date, description, _id, title, url = (
            itm[k] for k in sorted(required_keys)
        )

        authors = authors.split(", ")
        if key in ("publications", "applications"):
            authors = [author.split(" ")[-1] for author in authors]
        authors = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")

        if "authors_url" in itm:
            authors = f"[{authors}]({itm['authors_url']})"

        md_str = f"1. {date} - [{title}]({url}) by {authors}"

        indent = " " * 3

        if key in ("packages", "code") and url.startswith("https://github.com"):

            gh_login, repo_name = url.split("/")[3:5]
            md_str += (
                f'\n{indent}&ensp;<img src="https://img.shields.io/github/stars/'
                f'{gh_login}/{repo_name}" alt="GitHub repo stars" valign="middle" />'
            )

        description = description.removesuffix("\n").replace("\n", f"\n{indent}> ")
        description = re.sub(r"\s+\n", "\n", description)  # remove trailing whitespace
        md_str += f"\n\n{indent}> {description}"
        if repo := itm.get("repo", None):
            md_str += f" [[Code]({repo})]"

        sec["markdown"] += md_str + "\n\n"


# look ahead without matching
start_section_pat = lambda title: f"(?<={title}\n\n)"
# look behind without matching
next_section_pat = "(?=<br>\n\n## )"


with open(f"{ROOT}/readme.md", "r+") as file:

    readme = file.read()

    for sec in sections.values():
        section_start = start_section_pat(sec["title"])

        # match everything up to next heading
        readme = re.sub(
            rf"{section_start}[\s\S]+?\n\n{next_section_pat}", sec["markdown"], readme
        )

    file.seek(0)
    file.write(readme)
    file.truncate()

section_counts = "\n".join(
    f"- {key}: {len(sec['items'])}" for key, sec in sections.items()
)
print(f"finished writing {len(seen_ids)} items to readme:\n{section_counts}")
