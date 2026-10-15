---
name: qrcad-style
description: >
  Apply the QRCAD document style when writing any technical paper, report, or
  reference document in the style of CAD_Descriptors_Explained.pdf,
  CAD_Descriptors_Complexity_Analysis.pdf, or CAD_GlassBox_ML_Options.pdf.
  Trigger whenever the user asks to write a technical document, paper, or
  companion doc "in the same style" as these PDFs, or references QRCAD
  formatting, or uploads one of these documents and asks to produce something
  similar. Also trigger for any new section, descriptor write-up, or algorithm
  block that should match the established house style.
sources: [chat]
aliases: [qrcad-format, companion-doc-style, QRCAD style]
---

# QRCAD Document Style Guide

Extracted from three companion documents produced for the QRCAD project
(CAD_Descriptors_Explained.pdf, CAD_Descriptors_Complexity_Analysis.pdf,
CAD_GlassBox_ML_Options.pdf). Every rule below is derived from those sources,
not invented.

---

## 1. Cover Page

Four elements in this order, no headers, no extra decoration:

```
[Title — short noun phrase, title case]
[Subtitle — sentence-case phrase after an em-dash, or two lines joined by a comma]

[Project tag — "QRCAD: <one-line project description>"]
[Companion line — "Companion to <Filename.pdf>" — omit if this is the primary doc]
[Author/recipient — "Prepared for <Full Name>, <Dept.>, <Institution>"]
[Date — Day Month Year (e.g., 23 September 2026)]
```

Then immediately one abstract paragraph (no "Abstract" header).

**Abstract paragraph rules:**
- 3–6 sentences, no bullets
- Sentence 1: states what the document covers and its scope
- Sentence 2–4: lists the main areas or structure, specifically
- Final sentence: states a concrete scope boundary or a specific quantitative
  caveat (e.g., "Nothing here has yet been run on the 600-model dataset.")
- Never end with a forward-looking promise ("this will demonstrate…")
- Use "Gains are estimates unless marked Checked" or similar epistemic flags
  when the document contains projections

---

## 2. Table of Contents

Plain "Contents" header (not bold, not numbered).

Format per entry:
```
N  Section Title  . . . . . . . . . . . . . . page
   N.1  Subsection  . . . . . . . . . . . . . page
```

Rules:
- Every section and subsection appears; sub-subsections (N.M.P) are not listed
- Page numbers right-aligned with dot leaders
- No bold on any TOC line

---

## 3. Heading Hierarchy

| Level | Format | Example |
|-------|--------|---------|
| H1 | Bold + number | **3 Shared preprocessing** |
| H2 | Bold + decimal number | **3.4 Area-weighted surface sampling** |
| H3 | Italic inline label (not a heading) | *In plain English* |

H1 and H2 are the only true headings. H3 is a bold or italic inline callout
within a section, never given its own line in the TOC.

Sentence case everywhere: capitalise only the first word and proper nouns.

---

## 4. Within-Section Structural Patterns

### 4.1 Descriptor section (used in Explained.pdf)

Each descriptor follows this template exactly, in this order:

```
[Descriptor name in heading]
vector length [N]
Depends on: [P-step list]

In plain English
[3–5 sentence prose explanation using a concrete mechanical analogy]

Mathematics.
[Inline math block; no separate "Math" header]

Algorithm N  [Descriptor Name]
Input: [list]
1: step
2: step
3: return [output]

Why it is needed: [the baseline problem]
[Part A vs Part B table or prose comparison]

What it still cannot see
[Specific limitation, stated as a concrete geometric case]
```

**Inline tags:**
- `vector length 32` — flush left, before the body, no colon
- `Depends on: P1, P4` — flush left, right after the vector length tag

**"In plain English" block:**
- Starts with a concrete analogy using a named physical object (ball, shaft,
  hollow ball, gear)
- States what the descriptor measures in plain English before showing any math
- No "it is worth noting" or "importantly"

**Algorithm block:**
- Header: `Algorithm N  [Name]`
- `Input:` line explicit
- Steps numbered; each is one line
- `return` statement always last

**"Why it is needed" block:**
- Names exactly two parts (Part A, Part B) or uses a table
- Shows which earlier descriptors cannot separate them and why
- Ends with the SNR value or a "same by construction" justification

**"What it still cannot see" block:**
- One concrete geometric counterexample
- No abstract summary sentence

### 4.2 Complexity section (used in Complexity_Analysis.pdf)

Each descriptor sub-section follows this template:

```
[N.M  Descriptor name]
vector length [N]  Depends on: [P-step list].

Step table:
| Step | Time | Space |
...

Result
Time [formula]; space [formula]. About [N] × 10^M operations at defaults.

Measured
Exponent [X.XX] (predicted [Y]). [T] s and [M] MB at [size].
[Any notable sub-variant or warning.]
```

**Result callout:**
- Bold "Result" on its own line, no colon, immediately followed by the
  formula statement
- Always ends with "About [N] × 10^M operations at defaults."
- States both time and space in one sentence

**Measured callout:**
- Bold "Measured" on its own line
- First clause: "Exponent X.XX (predicted Y)."
- Second clause: time and memory at default sizes
- Optional third clause: comparison to a naive alternative (e.g., "27× slower")

**Step table:**
- Three columns: Step | Time | Space
- Step column is descriptive prose (not a name); can include the flop count
- Time and Space use Θ / O notation with parenthesised content
- Two Space entries when vectorised and streamed differ: write as
  "Θ(S) vect., Θ(B) streamed"

### 4.3 Options section (used in GlassBox_ML_Options.pdf)

Each option uses a short code (S1, H2, L3, F4) as an inline label before the
title. Format:

```
N.M  [Code]: [Option name]

[Tier: T0 / T1 / T1-calibrated / T2 / T3]
[Expected gain estimate; mark "Checked" only if empirically verified]

[Body: math, algorithm, cost]
```

Tier table (use this exact structure when defining tiers):

| Tier | Kind | What it may contain |
|------|------|---------------------|
| T0 | fixed math | closed-form geometry |
| T1 | learned numbers | named numbers fitted by explicit objective |
| T1 | calibrated | threshold as quantile of measured distribution |
| T2 | grey box | neural network whose outputs are named quantities |
| T3 | fenced embedding | opaque learned vector allowed as one extra block |

---

## 5. Prose Style Rules

### What to do
- Use present tense throughout ("LBE dominates", "the cap bounds")
- Give specific numbers: "1743× faster", "about 3–4×", "0.018%"
- Reference sections with § in parentheses: (§3.4), (§7.2)
- State the bottleneck first, then the fix
- One sentence per finding; lead with the number

### What never to write

| Banned phrase | Replace with |
|---------------|-------------|
| "it is worth noting that" | just state the fact |
| "it is important to" | just state why |
| "in summary" / "in conclusion" | cut; let the finding close the section |
| "this underscores the need for" | state the need directly |
| "a unified framework" | name the specific thing |
| "coordinated advances in" | name the specific advance |
| "further work is needed" | state what specifically or omit |
| three-item enumerations ("fast, robust, and accurate") | choose the one that matters |

**Never end a section with a convergence/outlook sentence.** End with the last
quantitative finding.

### Sentence rhythm
Vary length deliberately. Short declarative sentences (≤ 10 words) carry key
findings. Long sentences (20–35 words) carry derivations or conditionals. No
three consecutive sentences of the same length.

---

## 6. Typography and Notation

### Numbers and math
- Scientific notation with × and spaces: `2 × 10^4`, not `2e4` or `20000`
- Ranges use en-dash, no spaces: `77–78 MB`, `1–1.5`, `0.03–0.3 s`
- Multiplication speedup: `27×` (no space before ×)
- Percentages: `0.018%` (full precision)
- Θ and O notation: always in math (Θ(F), O(n log n))
- Subscripts on complexity: `V^{3/2}`, `n^2`

### Variables
- Single-letter variables: italic in text (*F*, *n*, *S*, *B*, *V*)
- Named constants in symbol table (see §1.2 of Complexity doc for format)
- Greek: Θ, α, ε, φ spelled out in LaTeX; in prose write "epsilon" or use the
  symbol directly

### Symbol table format
Whenever a document introduces ≥ 5 symbols, include a two-column table:

| Symbol | Meaning | Default |
|--------|---------|---------|
| F | triangles in the mesh | 5 × 10^4 |
| n | surface sample pool | 2 × 10^4 |

Always include a "Default" column with the value used in experiments.

### Code terms
- File names: `cache.py`, `SKILL.md` (code font)
- Library names: plain text first use, code on second (NumPy → `np.unique`)
- Algorithm names: plain bold (**SHPS**, **LBE**)

### Footnotes
Use †, ‡, § symbols (in that order). Footnote text is compact and specific:
- "† H0 union–find core only, as a Python loop."
- "‡ Free-form part with the N_fam ≤ 64 cap; a prismatic box takes 0.0126 s."

---

## 7. Tables

Standard table rules:
- Header row in bold or separated by a heavier rule
- Columns: left-align text, right-align numbers
- Units in the column header, not in each cell ("Measured [s]", not "0.26 s/model")
- No vertical rules except in complexity step tables
- Include "Notes" column when a cell needs a qualifier rather than a footnote

**Complexity summary table** (the full-pipeline table in §2.1) has these columns
in this order: Step | Time | Space | Ops | Measured | exp. (pred.)

**Scaling comparison** (§6.2 style): Step | Varied | Predicted | Measured | Comment

---

## 8. Summary / Findings Section

Place immediately after the TOC (before the main body) or as §2 in the document.
Title: "Summary of results" or "Summary" (never "Executive summary").

Always include a "N findings that matter for implementation" numbered list.
Each finding:
- Bold title (≤ 6 words), no period
- One sentence body: the finding stated as a specific quantitative fact or
  a concrete named algorithm/technique
- No sub-bullets

Example format:
```
1. The exact diameter is quadratic on rounded parts. On curved surfaces h = Θ(n),
   so comparing all hull pairs is O(n²): 5.22 s, more than all fifteen descriptors
   combined. The double-normal iteration gives the same value to 0.018% in 0.003 s.
```

---

## 9. Cost/Performance Section (Ranked by Payoff)

Title exactly: "Reducing cost: ranked by payoff" (or "Build order", "Practical
order of adoption").

Format: numbered list, each item opens with the technique in bold, then one
sentence of context, then the gain:

```
1. [Technique]: [what it replaces]. [Gain stated as a ratio or absolute saving].
2. ...
```

No sub-bullets. No hedging ("may improve", "could help"). State the gain or
note it is an estimate.

---

## 10. Checklist Before Finalising

Run through these before submitting any document in this style:

- [ ] Cover page has all four elements (title, project tag, recipient, date)
- [ ] Abstract ends with a concrete factual statement, not a forward promise
- [ ] Every section and subsection is in the TOC
- [ ] Every descriptor has vector length + dependency tags
- [ ] Every Result block ends with "About N × 10^M operations at defaults."
- [ ] Every Measured block states exponent, time, and memory
- [ ] No banned phrases from §5
- [ ] No section ends with a convergence/outlook sentence
- [ ] Ranges use en-dash, speedups use ×, sci-notation uses ×
- [ ] Symbol table present if ≥ 5 symbols introduced
- [ ] Footnote symbols in order: †, ‡, §
