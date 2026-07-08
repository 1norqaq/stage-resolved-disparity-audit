# Anonymization Notes for Double-Anonymous Review

The review repository should be checked before submission to ensure that it contains no identifying information.

## Remove before submission

- author names and affiliations;
- employer name, locations, domains, or internal project names;
- private file paths and usernames;
- API keys, tokens, credentials, and cloud bucket names;
- Git remotes tied to identifiable accounts;
- commit history if it contains identifying metadata;
- comments referring to internal teams, vendors, or deployment IDs;
- raw data and linkage keys.

## Suggested repository preparation

```bash
git init
find . -name "*.csv" -o -name "*.parquet" -o -name "*.pkl"
grep -R "@" . --exclude-dir=.git
grep -R "http" . --exclude-dir=.git
```

Before submitting a link, inspect the repository as a fresh clone and verify that all links, issue trackers, package metadata, and file histories are anonymous.
