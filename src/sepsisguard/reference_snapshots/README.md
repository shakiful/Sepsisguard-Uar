# Fixed sepsis reference corpus — 2026-11-08

Source: CDC, **About Sepsis**, and NIH NIGMS, **Sepsis**. The source URLs, final redirected URLs, retrieval timestamps and SHA256 hashes are recorded in `corpus.json`. Paragraph/list text is extracted without substantive rewriting; whitespace is normalized. Images and logos are excluded. This is a small educational corpus for retrospective explanation experiments, not a comprehensive clinical guideline collection.

The agency text is available free of charge from the linked original pages. This project is independent; use of these materials does not imply endorsement by CDC, NIH, HHS or the United States Government. Check the original pages for current information. The fixed version is used to make experiments reproducible and avoid live-site HTTP failures in Colab.

- CDC source: https://www.cdc.gov/sepsis/about/index.html
- NIH NIGMS source: https://www.nigms.nih.gov/education/fact-sheets/Pages/sepsis.aspx
- CDC reuse policy: https://www.cdc.gov/other/agencymaterials.html
- NIH public-domain guidance: https://www.nih.gov/about-nih/frequently-asked-questions

`build_corpus()` installs this verified snapshot on a fresh run without HTTP. Existing verified private snapshots remain pinned. To intentionally fetch a new live version, call `build_corpus(path, refresh=True)` and record a new experiment identity; that operation may fail if the website disallows server-side requests. Never replace failed evidence with synthetic or invented text.
