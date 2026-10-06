# Local preview only. GitHub Pages builds the site server-side from `_config.yml`, so this file is
# not required for deployment — it exists so `bundle exec jekyll serve` reproduces that build
# locally rather than approximating it.
#
#   bundle install
#   bundle exec jekyll serve                   # http://127.0.0.1:4000
#
# No --baseurl override needed: this is a user site, so the committed baseurl is already empty and
# local paths match deployed ones.
#
# If bundler refuses to resolve, the cause is usually macOS's /usr/bin/ruby (2.6), which the
# github-pages gem no longer supports — install a current Ruby (`brew install ruby` or rbenv) and
# retry. `_tools/validate.py` covers the structural failures without needing Ruby at all, but it is
# a pre-flight, not a substitute for the real build.
source "https://rubygems.org"

# The metagem that pins Jekyll and every plugin to the exact versions GitHub Pages runs, so a page
# that builds here builds there. Pinning jekyll directly instead is how local and deployed drift.
gem "github-pages", group: :jekyll_plugins
