# Paper Versions Archive

This directory contains the historical versions of the manuscript associated with:

**“A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds”**

The goal of this archive is to preserve the evolution of the scientific document across major revisions without cluttering the main `paper/` directory. Each version is stored in its own dedicated folder and contains the compiled manuscript along with optional notes explaining the changes or providing additional context.

---

## 📁 Structure

```
paper_versions/
    latest/                 # most recent manuscript version
    v1 ... v(N-1)/          # all previous numbered versions
```

* **`latest/`**
  Always contains the **most recent** version of the manuscript. This folder may also include notes documenting changes, explanations of conceptual updates, or additional reference files related to the active version.

* **Numbered folders (`v1/`, `v2/`, …)**
  Contain older versions of the manuscript, each isolated with its corresponding PDF and any optional notes for that specific version.

---

## 📝 Versioning Policy

* Every major scientific revision receives its own folder (`v3/`, `v4/`, …).
* Folders may contain:

  * the compiled manuscript,
  * optional notes (`revision_notes.md`, `changes.txt`, etc.),
  * supporting materials relevant to that version.
* The **working version** of the paper always lives only in the main `paper/` directory.
* Version numbers are assigned at the folder level, **not** inside the manuscript filename.

---

## 🎯 Purpose

This archive:

* preserves the scientific development of the project,
* maintains reproducibility for citations and historical reference,
* documents conceptual transitions between versions,
* isolates legacy documents from the current active manuscript.

---

If new versions are produced, simply create a new folder following the same conventions (`v3/`, `v4/`, …).
