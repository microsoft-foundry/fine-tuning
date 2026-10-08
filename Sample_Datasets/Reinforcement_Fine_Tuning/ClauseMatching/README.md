# Notice

This is a sample training and validation data for RFT, along with sample graders. This was curated using [this](https://zenodo.org/records/4595826) as the reference dataset.

## Dataset Summary

Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of 13,000+ labels in 510 commercial legal contracts that have been manually labeled under the supervision of experienced lawyers to identify 41 types of legal clauses that are considered important in contract review in connection with a corporate transaction, including mergers & acquisitions, etc.
CUAD is curated and maintained by The Atticus Project, Inc. to support NLP research and development in legal contract review. Read the full CUAD v1 announcement [here](https://www.atticusprojectai.org/cuadv1-announcement)!

## Graders
The repo contains jsonl specifications for two graders - a *multigrader*, combinining string check and text similarity, and a model based grader.
