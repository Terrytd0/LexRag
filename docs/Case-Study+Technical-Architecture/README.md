# LexRAG — Case Study & Technical Architecture

Two companion documents built with plain semantic HTML and CSS (no frameworks, no JavaScript), following the same pattern used across this portfolio's projects.

## Folder contents

```
Case-Study+Technical-Architecture/
├── case-study.html                          # Business-facing narrative: problem → solution → evidence
├── case-study.md                             # Markdown version (GitHub rendering, ATS-style text)
├── LexRAG-Case-Study.pdf                      # Print-exported PDF
├── technical-architecture.html                # Engineering-facing reference: components, data flow, gaps
├── technical-architecture.md                   # Markdown version
├── LexRAG-Technical-Architecture.pdf            # Print-exported PDF
├── styles.css                                  # Shared styling (screen + print/A4 rules), both documents
├── assets/                                     # Architecture diagram referenced by both documents
└── README.md
```

## What each document is for

- **Case Study** — the story: the problem (legal research needs evidence-grounded answers, not confident guesses), the solution (hybrid retrieval + reranking + citation-grounded refusal-aware generation), and real measured evaluation results — including the ones that don't yet clear the target bar. A **Resources** section at the bottom links to the GitHub repo, README, evaluation notes, and docs folder.
- **Technical Architecture** — the engineering reference: system architecture, per-capability acceptance status against the project's own documented thresholds, data flow, error handling, testing/CI, and an explicit known-limitations list.

## Why this project leans on numbers instead of screenshots

LexRAG's README documents a real golden-dataset evaluation harness (RAGAS + DeepEval) with measured recall, precision, faithfulness, refusal accuracy, and latency numbers — including a CI quality gate that has been demonstrated both failing and passing against real runs. Both documents lean on that quantified evidence rather than UI screenshots, since it's the strongest, most verifiable evidence this project has. Two acceptance criteria (100% negative-case refusal, P95 latency ≤ 6s) are stated as currently **not met** in both documents, matching the project's own README.

## How to edit

Content lives directly in `case-study.html` / `technical-architecture.html` as semantic `<section>` blocks; the `.md` files are the plain-text equivalents kept in sync by hand. Shared visual language lives in `styles.css` (identical design system to the other case studies in this portfolio) — edit content in the HTML files, not the CSS, for normal changes.

The architecture diagram (`assets/LexRAG.png`) is unusually tall (a multi-panel pipeline diagram), so both HTML files wrap its `<figure>` in a `diagram-tall` class that caps its rendered height at 620px — without this, the image spans 2–3 mostly-blank PDF pages when printed. Keep that class on the figure if the diagram image is ever swapped for a similarly tall one.

## Exporting to PDF

Both PDFs were generated headlessly via the Chrome DevTools Protocol (`Page.printToPDF` with `displayHeaderFooter: false`) rather than the plain `chrome --headless --print-to-pdf` CLI flag, which on this machine always injected a browser-default header/footer regardless of flags passed. To regenerate after edits, open either HTML file in a Chromium-based browser and use **Print → Save as PDF** (the stylesheet sets A4 size and margins via `@page`, headers/footers off), or drive `Page.printToPDF` over the DevTools Protocol.

## Design notes

Same visual system as this portfolio's other case studies: dark-navy headings and table headers, a blue accent for section numbers/links, tinted stage/card backgrounds to distinguish pipeline steps, and green/amber/red status pills for acceptance status. Resources/verification links are isolated at the bottom of each document, not woven into the narrative.
