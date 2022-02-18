from __future__ import annotations

import re
from typing import cast

import yaml


sections: dict[str, dict[str, str | list[dict[str, str]]]] = {
    "publications": {"title": "## 📝 Publications"},
    "applications": {"title": "### 🛠️ Applications"},
    "videos": {"title": "## 📺 Videos"},
    "packages": {"title": "## 📦 Packages"},
    "code": {"title": "## 🧑‍💻 Code"},
    "posts": {"title": "## 🌐 Blog Posts"},
}

for key in sections:
    with open(f"data/{key}.yml") as file:
        sections[key]["items"] = yaml.safe_load(file.read())


seen_ids: set[str] = set()


req_keys = "id,title,url,date,authors,description".split(",")
opt_keys = ["org", "authorsUrl", "for"]


def validate_item(itm: dict[str, str]) -> None:
    itm_keys = list(itm.keys())
    err = None

    if (id := itm["id"]) in seen_ids:
        err = f"Duplicate id: {id}"
    else:
        seen_ids.add(id)

    if not id.startswith(("pub-", "app-", "vid-", "pkg-", "code-", "post-")):
        err = f"Invalid id: {id}"

    valid_fors = ("PyTorch", "TensorFlow", "JAX", "Julia", "Others")
    if id.startswith(("pkg-", "code-")) and itm["for"] not in valid_fors:
        err = f"Invalid for in {id}: {itm['for']}, must be one of {valid_fors}"

    if missing_keys := [k for k in req_keys if k not in itm_keys]:
        err = f"Missing key(s) in {id}: {missing_keys}"

    if bad_keys := set(itm_keys) - set(req_keys + opt_keys):
        err = f"Unexpected key(s) in {id}: {bad_keys}"

    if err:
        raise ValueError(err)


for key, sec in sections.items():
    sec["markdown"] = ""

    for idx, itm in enumerate(sec["items"], 1):
        itm = cast(dict[str, str], itm)
        validate_item(itm)

        title, url, date, authors, description = (itm[k] for k in req_keys[1:])

        authors = authors.split(", ")
        authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

        if "authorsUrl" in itm:
            authors = f"[{authors}]({itm['authorsUrl']})"

        md_str = f"{idx}. {date} - [{title}]({url}) by {authors}"

        indent = len(f"{idx}. ") * " "
        if key in ("packages", "code") and url.startswith("https://github.com"):
            repo_handle = "/".join(url.split("/")[-2:])
            md_str += (
                f'\n{indent}&ensp;<img src="https://img.shields.io/github/stars/{repo_handle}" '
                'alt="GitHub repo stars" valign="middle" />'
            )

        description = description.replace("\n", f"\n{indent}> ")
        md_str += f"\n\n{indent}> {description}"

        sec["markdown"] += md_str + "\n\n"


# look ahead without matching
start_section_pat = lambda title: f"(?<={title}\n\n)"
# look behind without matching
next_section_pat = "(?=<br>\n\n##{1,5} )"

with open("readme.md", "r") as file:
    readme = file.read()

with open("readme-test.md", "r+") as file:

    # readme = file.read()

    for key, val in sections.items():
        section_start = start_section_pat(val["title"])

        # match everything up to next heading
        readme = re.sub(
            rf"{section_start}[\s\S]+?\n\n{next_section_pat}", val["markdown"], readme
        )

    file.seek(0)
    file.write(readme)
    file.truncate()
