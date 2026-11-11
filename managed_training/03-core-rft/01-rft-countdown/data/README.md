# Countdown data

`preserved/` contains the canonical LF-normalized Countdown datasets. The notebook trains with the 100-row training and 50-row validation files. The 100-row evaluation file is validated as provenance but is not uploaded or run.

`hashes.json` records canonical LF byte counts and SHA-256 values for its listed artifacts. Notebook guards normalize CRLF to LF before validation and upload so Windows checkout settings cannot change the service input contract. The scoped `.gitattributes` preserves LF endings for the canonical 100/50/100 datasets.
