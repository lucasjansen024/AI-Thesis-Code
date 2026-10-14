# Paper drafts

| File | Content |
|---|---|
| `01_introduction.md` | Introduction draft |
| `02_methods.md` | Methods draft (full version; main-text and supplement split marked) |
| `03_results.md` | Results draft v0.1: Aim 1 in full, Aim 2 **interim** (2b Step pending); open points for co-authors at the end |
| `references.md` | Working reference list with verification status |

Source of truth for every methodological fact: [`../METHODOLOGY.md`](../METHODOLOGY.md). Update that first, then the drafts.

## Target journals

**Journal of Sport and Health Science (JSHS)** or **Sports Medicine**. Not decided. Check the current author guidelines before formatting: the points below come from a quick web search (9 Oct 2026) and from general knowledge, and must be verified.

| | JSHS (Elsevier) | Sports Medicine (Springer) |
|---|---|---|
| Length | Original Article ≤ 5,000 words, normally ≤ 6 display items (tables + figures) | verify; original research is accepted but must have "a strong link to clinical practice in sport and exercise medicine" |
| Extras | verify (highlights? structured abstract?) | **Key Points** box: ~3 short stand-alone statements after the abstract |
| References | numbered (verify) | numbered (verify) |

## What this means for the writing

1. **Audience is sports/exercise medicine, not machine learning or applied physiology.** The introduction leads with the practical problem (monitoring internal load, exercise prescription, rehabilitation) and explains the model in plain terms. Machine-learning details belong in the methods or the supplement, and each one should be justified by what it protects against: leakage, optimism, unfair comparisons.
2. **Word budget (JSHS).** About 5,000 words for the whole article means about 1,500–2,000 words of main-text methods. `02_methods.md` is deliberately the long version. Each section is tagged **[main]** or **[supplement]** to show what moves to *Supplementary Methods* when it is cut. Keep the long version as the supplement, not in the bin.
3. **Display items (JSHS ≤ 6).** The methods use two main-text items: **Table 1** (participants) and **Figure 1** (protocols + analysis pipeline). Protocol timelines, inclusion, input domains, training settings and the 46 configurations are supplementary tables (S1–S5). The remaining 3–4 items are left for the results.
4. **Practice link (Sports Medicine).** Make explicit what practitioners gain. That includes which sensors are worth wearing, what accuracy to expect without a power meter, and what the error means in practice (e.g. relative to the difference between intensity domains). Draft Key Points once the results exist.
5. **Citations** are `[Author Year]` placeholders for now; both journals use numbered styles, so convert when the journal is chosen.
