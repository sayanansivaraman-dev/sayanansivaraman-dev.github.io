"""Pre-flight checks on the Jekyll site, for when there is no local Ruby to build with.

    python3 _tools/validate.py

Catches the failures that otherwise reach a live 404 or a broken image: unparseable front matter,
unbalanced Liquid, an asset path that does not resolve, an internal link to a page that does not
exist. GitHub Pages is the real build; this is what you run before pushing.

Also lists remaining `[[placeholder]]` blocks, which are intentional drafting markers rather than
errors -- the count going to zero is the signal a page is finished.
"""

import re
import sys
from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parent.parent
problems = []

# --- _config.yml parses, and baseurl is consistent with the repo name -------
config = yaml.safe_load((SITE / "_config.yml").read_text())
print(f"config    : baseurl={config['baseurl']!r} url={config['url']!r}")

# --- front matter on every content page -------------------------------------
# Honouring `exclude`: a file Jekyll never processes needs no front matter, and flagging it is a
# false positive that trains you to ignore the output.
excluded = set(config.get("exclude", []))
pages = [
    p
    for p in sorted(SITE.glob("*.md")) + sorted(SITE.glob("projects/*.md"))
    if p.name not in excluded
]
for page in pages:
    text = page.read_text()
    if not text.startswith("---\n"):
        problems.append(f"{page.name}: no front matter")
        continue
    end = text.index("\n---\n", 3)
    try:
        front = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        problems.append(f"{page.name}: front matter is not valid YAML -- {exc}")
        continue
    print(f"page      : {page.relative_to(SITE)}  title={front.get('title')!r}")

# --- Liquid tags balanced ----------------------------------------------------
for page in pages + list(SITE.glob("_layouts/*.html")):
    text = page.read_text()
    for opener, closer, name in (("{{", "}}", "output"), ("{%", "%}", "tag")):
        if text.count(opener) != text.count(closer):
            problems.append(
                f"{page.name}: unbalanced Liquid {name} -- "
                f"{text.count(opener)} {opener} vs {text.count(closer)} {closer}"
            )

# --- every referenced asset exists ------------------------------------------
asset_re = re.compile(r"'(/assets/[^']+)'")
referenced = set()
for page in pages + list(SITE.glob("_layouts/*.html")):
    referenced |= set(asset_re.findall(page.read_text()))
for ref in sorted(referenced):
    target = SITE / ref.lstrip("/")
    status = "ok  " if target.is_file() else "MISS"
    if not target.is_file():
        problems.append(f"missing asset: {ref}")
    print(f"asset     : {status} {ref}")

# --- internal page links resolve --------------------------------------------
link_re = re.compile(r"'(/[^']*)'\s*\|\s*relative_url")
for page in pages + list(SITE.glob("_layouts/*.html")):
    for link in link_re.findall(page.read_text()):
        if link.startswith("/assets") or link == "/":
            continue
        slug = link.strip("/")
        candidates = [SITE / f"{slug}.md", SITE / slug / "index.md", SITE / slug]
        if not any(c.exists() for c in candidates):
            problems.append(f"{page.name}: internal link {link} resolves to nothing")
        else:
            print(f"link      : ok   {link}")

# --- unfilled placeholders ---------------------------------------------------
for page in pages:
    hits = len(re.findall(r"\[\[", page.read_text()))
    if hits:
        print(f"TODO      : {page.relative_to(SITE)} has {hits} [[placeholder]] block(s)")

# --- checks against the BUILT output, when it exists -------------------------
# Both of the bugs this section catches shipped and were found by eye in a browser: a card whose
# closing tag kramdown escaped into visible text, and a nav link pointing at the page it was on.
# Both are mechanically detectable, so they are checked here rather than noticed again.
site = SITE / "_site"
if not site.is_dir():
    print("\nbuilt     : _site/ absent -- run `bundle exec jekyll serve` for the rendered checks")
else:
    built = sorted(site.rglob("*.html"))
    print(f"\nbuilt     : {len(built)} page(s) in _site/")

    ids = {}
    for page in built:
        html = page.read_text()
        route = "/" + str(page.relative_to(site)).replace("index.html", "")
        ids[route.replace(".html", "/")] = set(re.findall(r'id="([^"]+)"', html))

        # Kramdown parses block-level raw HTML verbatim but treats INLINE tags (notably `<a>`) as
        # span content: it closes them immediately, hoists the children out, and escapes the
        # closing tag. A literal `&lt;/` in the output is that signature.
        escaped = re.findall(r"&lt;/\w+&gt;", html)
        if escaped:
            problems.append(
                f"{page.relative_to(site)}: {len(escaped)} escaped closing tag(s) "
                f"({escaped[0]}) -- an inline tag used as a block; wrap it in a <div>"
            )

        if "{{" in html or "{%" in html:
            problems.append(f"{page.relative_to(site)}: unrendered Liquid in the output")

    # Every same-site fragment link must resolve to an id on the page it targets, and must not
    # point at the page it sits on with no fragment at all.
    for page in built:
        html = page.read_text()
        here = "/" + str(page.relative_to(site)).replace("index.html", "").replace(".html", "/")
        for tag in re.finditer(r"<a\s([^>]*)>", html):
            attrs = tag.group(1)
            href_match = re.search(r'href="(/[^"]*)"', attrs)
            if not href_match:
                continue
            href = href_match.group(1)
            class_match = re.search(r'class="([^"]*)"', attrs)
            classes = class_match.group(1) if class_match else ""
            path, _, frag = href.partition("#")
            path = path or "/"
            if frag:
                target = ids.get(path if path.endswith("/") else path + "/", ids.get(path))
                if target is None:
                    problems.append(f"{page.relative_to(site)}: {href} -> unknown page")
                elif frag not in target:
                    problems.append(f"{page.relative_to(site)}: {href} -> no id {frag!r} there")
                else:
                    print(f"anchor    : ok   {here} -> {href}")
            elif path == here and not href.endswith((".css", ".png", ".gif")):
                # A header brand/logo link pointing home from the home page is universal
                # convention, not a dead link. Everything else self-linking is a mistake --
                # that is how the nav's "Projects" button pointed at the page it was on.
                if "home" in classes:
                    print(f"anchor    : ok   {here} -> {href} (brand link, self-link expected)")
                else:
                    problems.append(
                        f"{page.relative_to(site)}: {href} links to the page it is on -- "
                        "did you mean a #fragment?"
                    )

print()
if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("no structural problems found")
