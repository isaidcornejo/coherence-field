# Revision Notes – Version 2
Date: December 3, 2025  
Document: “A Scalar Diagnostic for Empirical Score Alignment on Fisher Manifolds”

This document summarizes the conceptual and structural changes introduced in
Version 2 of the manuscript relative to the original “Coherence Field φ(θ)”
formulation (Version 1).

---

## 1. Conceptual Reformulation

### 1.1 Replacement of the Coherence Field φ
Version 1 introduced the “coherence field” φ(θ) based on excess Fisher-norm
alignment. Version 2 generalizes this idea by:

- Replacing φ(θ) with the scalar diagnostic  
  **A(θ; q) = Tr(G⁻¹ C) – D**
- Introducing the **alignment operator H = G⁻¹ C**
- Using its eigenvalues {λᵢ} to characterize empirical deformation through  
  **A = Σᵢ (λᵢ − 1)**

This yields a fully invariant, coordinate-free diagnostic.

### 1.2 Rectified Amplitude ϕ
While φ(θ) from Version 1 measured excess alignment directly, Version 2 defines:

**ϕ(θ; q) = max( sqrt(A), 0 )**

This acts as an “excess-alignment amplitude,” isolating reinforcement modes (λᵢ>1).

---

## 2. Geometric and Spectral Structure

### 2.1 Alignment Operator
Version 2 introduces the mixed tensor:

**H = G⁻¹ C**

and demonstrates:

- H is diagonalizable
- All eigenvalues λᵢ are real and non-negative
- H is similar to the symmetric positive-semidefinite matrix G⁻¹/² C G⁻¹/²

### 2.2 Identity for A
The new spectral representation:

**A = Σ (λᵢ − 1)**

reveals:
- equilibrium (λᵢ=1),
- suppression (λᵢ<1),
- reinforcement (λᵢ>1).

This identity did not exist in Version 1.

---

## 3. Structural Improvements

### 3.1 Complete reorganization of the paper
- Added Background section on Fisher geometry  
- Introduced alignment tensor Δ  
- Added spectral operator section  
- Replaced φ-based derivation with scalar diagnostic A  
- Added alignment regimes (equilibrium, suppression, reinforcement)

### 3.2 New experimental section
Version 2 includes:

- Gaussian
- Laplace
- Gaussian Mixtures
- MNIST MLP (new)
- Summary table of regimes

All experiments now use a unified 5-step pipeline and share reproducible code.

---

## 4. Removed or Deprecated Material

### 4.1 Removed
- Variational field-theoretic formulation of φ  
- φ-based geometric flow discussion

### 4.2 Deprecated
- Interpretation of φ as a fundamental field  
- Non-spectral alignment measures

These remain in Version 1 for archival purposes.

---

## 5. Summary

Version 2 represents a **major conceptual transition**:
- From a geometric “coherence field φ(θ)”  
- To a **spectral, invariant diagnostic A(θ; q)** grounded in Fisher geometry.

This version constitutes the new baseline for subsequent theoretical development
and experimental extension.
