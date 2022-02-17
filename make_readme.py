from __future__ import annotations

import re

import yaml


sections = {
    "publications": {"title": "## 📝 Publications"},
    "applications": {"title": "### 🛠️ Applications"},
}

for key in sections:
    with open(f"{key}.yml") as file:
        sections[key]["items"] = yaml.safe_load(file.read())

for key, sec in sections.items():
    sec["markdown"] = ""

    for idx, itm in enumerate(sec["items"], 1):
        key_order = "id, title, url, date, authors, description"
        if key_order != (keys := ", ".join(itm.keys())):
            if len(key_order) == keys:
                err = "Wrong key order: "
            else:
                err = f"missing key(s), should have {key_order}"
            raise ValueError(f"{err}: {itm}")

        id, title, url, date, authors, description = itm.values()

        authors = authors.split(", ")
        authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

        pub_str = f"{idx}. {date} - [{title}]({url}) by {authors}"

        indent = len(f"{idx}. ") * " "
        description = description.replace("\n", f"\n{indent}> ")
        pub_str += f"\n\n{indent}> {description}"

        sec["markdown"] += pub_str + "\n\n"


# look ahead without matching
start_section_pat = lambda title: f"(?<={title}\n\n)"
# look behind without matching
next_section_pat = "(?=<br>\n\n##{1,5} )"


with open("readme-test.md", "r+") as file:

    readme = file.read()

    for key, val in sections.items():
        section_start = start_section_pat(val["title"])

        # match everything up to next heading
        readme = re.sub(
            rf"{section_start}[\s\S]+?\n\n{next_section_pat}", val["markdown"], readme
        )

    file.seek(0)
    file.write(readme)
    file.truncate()
