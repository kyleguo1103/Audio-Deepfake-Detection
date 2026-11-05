# DeepSpeak data restrictions and public-release boundary

DeepSpeak v2 is a gated dataset. This repository must not redistribute restricted media or reconstruct the restricted annotations.

## Excluded from the public repository

- DeepSpeak video or extracted audio;
- downloaded archive shards or range caches;
- raw DeepSpeak annotation CSV files;
- the frozen target manifest containing source filenames and identity mappings;
- acquisition ledgers and per-file hashes;
- XLS-R target embeddings;
- sample-level predictions containing DeepSpeak sample identifiers;
- local hard-case lists and identity-level error tables;
- Hugging Face tokens, credentials or cached authentication files.

## Included

- experiment code;
- target-selection rules and immutable dataset revision;
- model/configuration records;
- aggregate per-seed metrics;
- aggregate bootstrap intervals;
- figures and written interpretations;
- verification reports that do not reveal restricted sample identifiers.

The full sample-level evidence remains stored locally for supervisor review by authorised dataset users. If the dataset licence is later confirmed to allow redistribution of additional derived artefacts, they should be reviewed separately before publication.

