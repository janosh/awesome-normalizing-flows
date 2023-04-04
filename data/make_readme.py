"""Script to generate readme.md from data/*.yml files."""

import datetime
import re
from os.path import dirname
from typing import TypedDict

import yaml

ROOT = dirname(dirname(__file__))


class Item(TypedDict):
    """An item in a readme section like a paper or package."""

    title: str
    authors: str
    date: datetime.date
    lang: str
    url: str
    description: str
    authors_url: str | None
    repo: str | None
    date_added: datetime.date | None


class Section(TypedDict):
    """A section of the readme like 'Publications' or 'Packages'."""

    title: str
    items: list[Item]
    markdown: str


titles = dict(
    publications="## 📝 Publications",
    applications="## 🛠️ Applications",
    videos="## 📺 Videos",
    packages="## 📦 Packages",
    repos="## 🧑‍💻 Repos",
    posts="## 🌐 Blog Posts",
)


def load_items(key: str) -> list[Item]:
    """Load list[Item] from YAML file."""
    with open(f"{ROOT}/data/{key}.yml", encoding="utf8") as file:
        return yaml.safe_load(file.read())


sections: dict[str, Section] = {
    key: dict(title=titles[key], items=load_items(key), markdown="")
    for key in titles  # markdown is set below
}


seen_titles: set[tuple[str, str]] = set()
required_keys = {"title", "url", "date", "authors", "description"}
optional_keys = {"authors_url", "lang", "repo", "docs", "date_added", "last_updated"}
valid_languages = {"PyTorch", "TensorFlow", "JAX", "Julia", "Other"}
et_al_after = 2


def validate_item(itm: Item, section_title: str) -> None:
    """Check that an item conforms to schema. Raise ValueError if not."""
    # no need to check for duplicate keys, YAML enforces that
    itm_keys = set(itm)
    err = None

    if (title := itm["title"]) in seen_titles:
        err = f"Duplicate {title = }"
    else:
        seen_titles.add((title, section_title))

    if section_title in ("packages", "repos") and itm["lang"] not in valid_languages:
        err = (
            f"Invalid lang in {title}: {itm['lang']}, must be one of {valid_languages}"
        )

    if missing_keys := required_keys - itm_keys:
        err = f"Missing key(s) in {title}: {missing_keys}"

    if bad_keys := itm_keys - required_keys - optional_keys:
        err = f"Unexpected key(s) in {title}: {bad_keys}"

    authors = itm["authors"]
    if "et al" in authors or "et. al" in authors:
        err = (
            f"Incomplete authors in {title}: don't use 'et al' in {authors = }, list "
            "them all"
        )

    if not isinstance(itm["date"], datetime.date):
        err = f"Invalid date in {title}: {itm['date']}"

    if date_added := itm.get("date_added"):
        assert isinstance(date_added, datetime.date)
    if last_updated := itm.get("last_updated"):
        assert isinstance(last_updated, datetime.date)

    if err:
        raise ValueError(err)


for key, section in sections.items():
    # Keep lang_names inside sections loop to refill language subsections for each new
    # section. Used by both repos and Packages. Is a list for order and mutability.
    lang_names = ["PyTorch", "TensorFlow", "JAX", "Julia", "Other"]

    # sort first by language with order determined by lang_names (only applies to
    # Package and repos sections), then by date
    section["items"].sort(key=lambda x: x["date"], reverse=True)
    if key in ("packages", "repos"):
        section["items"].sort(key=lambda itm: lang_names.index(itm["lang"]))

    # add item count after section title
    # section["markdown"] += f"\n\n{len(section['items'])} items\n\n"

    for itm in section["items"]:
        if (lang := itm.get("lang")) in lang_names:
            lang_names.remove(lang)
            # print language subsection title if this is the first item with that lang
            section["markdown"] += (
                f'<br>\n\n### <img src="assets/{lang.lower()}.svg" alt="{lang}" '
                f'height="20px"> &nbsp;{lang} {key.title()}\n\n'
            )

        validate_item(itm, section["title"])

        authors = itm["authors"]
        date = itm["date"]
        description = itm["description"]
        title = itm["title"]
        url = itm["url"]

        author_list = authors.split(", ")
        if key in ("publications", "applications"):
            # only show people's last name for papers
            author_list = [author.split(" ")[-1] for author in author_list]
        authors = ", ".join(author_list[:et_al_after])
        if len(author_list) > et_al_after:
            authors += " et al."

        if authors_url := itm.get("authors_url"):
            authors = f"[{authors}]({authors_url})"

        md_str = f"1. {date} - [{title}]({url}) by {authors}"

        if key in ("packages", "repos") and url.startswith("https://github.com"):
            gh_login, repo_name = url.split("/")[3:5]
            md_str += (
                f'\n&ensp;\n<img src="https://img.shields.io/github/stars/'
                f'{gh_login}/{repo_name}" alt="GitHub repo stars" valign="middle" />'
            )

        md_str += "<br>\n   " + description.removesuffix("\n")
        if docs := itm.get("docs"):
            md_str += f" [[Docs]({docs})]"
        if repo := itm.get("repo"):
            md_str += f" [[Code]({repo})]"

        section["markdown"] += md_str + "\n\n"


with open(f"{ROOT}/readme.md", "r+", encoding="utf8") as file:
    readme = file.read()

    for section in sections.values():
        # look ahead without matching
        section_start_pat = f"(?<={section['title']}\n\n)"
        # look behind without matching
        next_section_pat = "(?=<br>\n\n## )"

        # match everything up to next heading
        readme = re.sub(
            rf"{section_start_pat}[\s\S]+?\n\n{next_section_pat}",
            section["markdown"],
            readme,
        )

    file.seek(0)
    file.write(readme)
    file.truncate()

section_counts = "\n".join(
    f"- {key}: {len(sec['items'])}" for key, sec in sections.items()
)
print(f"finished writing {len(seen_titles)} items to readme:\n{section_counts}")
