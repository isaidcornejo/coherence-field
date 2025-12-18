# Revision Notes – Version 3  
Date: December 18, 2025  
Document: “A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds”

This document summarizes the conceptual clarifications and minor structural
refinements introduced in Version 3 of the manuscript, following peer review
and internal consistency checks. Version 3 does not alter the core diagnostic,
theoretical framework, or experimental results introduced in Version 2.

---

## 1. Scope of the Revision

Version 3 is a **minor revision** focused on clarifying the *operational
interpretation* of the scalar diagnostic under finite-sample estimation.
No new theoretical objects are introduced, and no existing definitions are
modified.

Specifically:
- The scalar diagnostic **𝒜(θ; q)** remains unchanged.
- The alignment operator **H = G⁻¹C** remains unchanged.
- All experiments, figures, and numerical results remain unchanged.

---

## 2. Finite-Sample Resolution Clarification (New in Version 3)

### 2.1 Motivation

While the diagnostic 𝒜 is defined exactly at the population level, empirical
estimation from finite samples introduces unavoidable spectral fluctuations.
In high-dimensional settings, the accumulation of sub-threshold fluctuations
can artificially inflate scalar summaries if not interpreted carefully.

Version 3 makes this empirical limitation explicit.

---

### 2.2 Minimal Coherent Alignment

Version 3 introduces a **resolution-aware interpretive refinement**:

- A finite-sample resolution scale  
  **ε_N ∼ O(N⁻¹ᐟ²)**

- A minimally resolvable excess-alignment diagnostic  
  **𝒜_min(θ; q) = Σ_{λᵢ > 1 + ε_N} (λᵢ − 1)**

This quantity:
- Filters out eigenvalue deviations attributable to finite-sample noise
- Identifies only empirically *resolvable* reinforcement modes
- Is explicitly operational and **not** a geometric invariant

The original diagnostic 𝒜 remains the fundamental, invariant quantity.

---

### 2.3 Resolvable Coherent Amplitude

For empirical interpretation, Version 3 introduces the derived quantity:

**φ_res(θ) = max{ √𝒜_min(θ; q), 0 }**

This provides a finite-sample–robust amplitude that:
- Suppresses artificial inflation from accumulated noise
- Is particularly relevant in large neural networks
- Plays an interpretive role analogous to φ, without redefining it

Neither φ_res nor 𝒜_min replace the original diagnostic or amplitude.

---

## 3. Structural Changes to the Manuscript

### 3.1 New Subsection

A new subsection was added:

**Section IV.E — Finite-Sample Resolution and Minimal Coherent Alignment**

This subsection:
- Appears within the spectral structure section
- Does not modify preceding derivations
- Serves as a bridge between ideal theory and empirical measurement

---

### 3.2 Introduction Clarification

A single sentence was added to the Introduction to acknowledge the operational
distinction between formal alignment and resolvable coherent alignment.
No new notation or definitions were introduced in the Introduction.

---

## 4. No Changes to Experiments or Results

- No experiments were added, removed, or modified.
- No figures were changed.
- No hyperparameters, datasets, or models were altered.
- No new sensitivity analyses were introduced.

All experimental interpretations remain valid; Version 3 only clarifies how
small-amplitude effects should be read under finite-sample resolution.

---

## 5. Relation to Previous Versions

- **Version 1**: Introduced a coherence-field formulation φ(θ) with variational
  dynamics.
- **Version 2**: Replaced the field-based formulation with a scalar, invariant,
  spectral diagnostic 𝒜(θ; q). Established the alignment operator and spectral
  identity.
- **Version 3**: Retains Version 2 as the theoretical baseline and adds an
  operational layer clarifying finite-sample resolution and empirical
  interpretability.

---

## 6. Summary

Version 3 represents a **clarifying refinement**, not a theoretical extension.

- The invariant diagnostic 𝒜 remains the core object.
- Finite-sample effects are explicitly acknowledged and controlled.
- Empirical robustness is improved without introducing new assumptions.
- The manuscript is now fully aligned with reviewer expectations and ready for
  publication.

No further changes are planned for this manuscript.
