#!/bin/bash
# Grep the tracked files for anything that looks like personal or site-specific information before
# pushing: home paths, e-mail addresses, API tokens, cluster hostnames / accounts, and the current
# user's login name (taken from $USER, so it never appears in this file). Empty output = clean.
cd "$(dirname "$0")/.."
PATTERNS='/home/|/global/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|lbl\.gov|lrc-|\.es1\b|pc_[a-z]+'
[ -n "${USER:-}" ] && PATTERNS="$PATTERNS|\\b$USER\\b"
[ -n "${PII_EXTRA:-}" ] && PATTERNS="$PATTERNS|$PII_EXTRA"   # e.g. PII_EXTRA='surname|firstname'
git ls-files -z | grep -zv '^scripts/check_pii.sh$' | xargs -0 grep -nIE "$PATTERNS" || echo "clean: no matches"
