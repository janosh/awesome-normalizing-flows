import datetime
import re
from os.path import dirname
from typing import TypedDict

import yaml

ROOT = dirname(dirname(__file__))


class Item(TypedDict):
    id: str
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
    title: str
    items: list[Item]
    markdown: str


titles = dict(
    publications="## 📝 Publications",
    applications="## 🛠️ Applications",
    videos="## 📺 Videos",
    packages="## 📦 Packages",
    code="## 🧑‍💻 Code",
    posts="## 🌐 Blog Posts",
)

sections: dict[str, Section] = {
    key: dict(
        title=titles[key],
        items=yaml.safe_load(open(f"{ROOT}/data/{key}.yml").read()),
        markdown="",  # will be filled below
    )
    for key in titles
}


seen_ids: set[str] = set()
required_keys = {"id", "title", "url", "date", "authors", "description"}
optional_keys = {"authors_url", "lang", "repo", "date_added", "last_updated"}
valid_languages = {"PyTorch", "TensorFlow", "JAX", "Julia", "Others"}
et_al_after = 2


def validate_item(itm: Item) -> None:
    """Checks that an item conforms to schema. Raises ValueError if not."""
    # no need to check for duplicate keys, YAML enforces that
    itm_keys = set(itm)
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

    if date_added := itm.get("date_added"):
        assert isinstance(date_added, datetime.date)
    if last_updated := itm.get("last_updated"):
        assert isinstance(last_updated, datetime.date)

    if err:
        raise ValueError(err)


for key, section in sections.items():
    # Keep lang_names inside sections loop to refill language subsections for each new
    # section. Used by both Code and Packages. Is a list for order and mutability.
    lang_names = ["PyTorch", "TensorFlow", "JAX", "Julia", "Others"]

    # sort first by language with order determined by lang_names (only applies to
    # Package and Code sections), then by date
    section["items"].sort(key=lambda x: x["date"], reverse=True)
    if key in ("packages", "code"):
        section["items"].sort(
            key=lambda itm: lang_names.index(itm["lang"])  # noqa: B023
        )

    # add item count after section title
    # section["markdown"] += f"\n\n{len(section['items'])} items\n\n"

    for itm in section["items"]:
        if (lang := itm.get("lang", None)) in lang_names:
            lang_names.remove(lang)
            # print language subsection title if this is the first item with that lang
            section["markdown"] += (
                f'<br>\n\n### <img src="assets/{lang.lower()}.svg" alt="{lang}" '
                f'height="20px"> &nbsp;{lang} {key.title()}\n\n'
            )

        validate_item(itm)

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

        if authors_url := itm.get("authors_url", None):
            authors = f"[{authors}]({authors_url})"

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

        section["markdown"] += md_str + "\n\n"


with open(f"{ROOT}/readme.md", "r+") as file:

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
print(f"finished writing {len(seen_ids)} items to readme:\n{section_counts}")
