# sayanansivaraman-dev.github.io

Source for **<https://sayanansivaraman-dev.github.io>** — a portfolio of computer-vision and
machine-learning write-ups.

Jekyll on GitHub Pages, no theme: three templates and one stylesheet, so there is nothing to
override. Pages are markdown; `projects/*.md` picks up the project layout automatically via
`_config.yml`.

## Local preview

```bash
bundle install
bundle exec jekyll serve      # http://127.0.0.1:4000
```

`baseurl` is empty because this is a *user* site (the repo is named `<username>.github.io`), so
local paths match deployed ones and no `--baseurl` override is needed. If this ever becomes a
project site, set `baseurl: /<repo-name>` in `_config.yml` and change nothing else — every link and
asset goes through Jekyll's `relative_url` filter rather than hard-coding a path.

macOS system Ruby is 2.6, which the `github-pages` gem no longer supports. If bundler refuses to
resolve, install a current Ruby (`brew install ruby`, or rbenv).

## Pre-flight checks

```bash
python3 _tools/validate.py
```

Needs no Ruby. Catches the failures that otherwise reach a live 404 or a broken image:
unparseable front matter, unbalanced Liquid, assets that do not resolve, internal links to pages
that do not exist. When `_site/` is present it also checks the rendered output for escaped closing
tags (the signature of an inline tag used as a block — kramdown silently mangles those), links
pointing at the page they sit on, and `#fragment` targets that do not exist.

It also counts remaining `[[placeholder]]` blocks, which are drafting markers rather than errors.
Zero is the signal a page is finished.

## Layout

```
_config.yml           baseurl, markdown settings, per-path layout defaults
_layouts/             default.html (shell, nav, Open Graph) · project.html (header, meta, refs)
_tools/validate.py    pre-flight checks; excluded from the build
assets/css/style.css  the whole stylesheet
assets/<project>/     figures, one directory per write-up
index.md              landing page
projects/*.md         one write-up each
```

## Licence

Site content © Sayanan Sivaraman. Figures and data derived from
[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) are licensed
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) and carry attribution.
