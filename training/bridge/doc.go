// Package bridge re-expresses a teacher's next-token distribution over a
// student vocabulary that tokenizes the same bytes differently, by
// Byte-Prefix Marginalization (BPM, "Cross-Tokenizer On-Policy Distillation
// via Byte-Prefix Marginalization", arXiv 2607.22334).
//
// Both vocabularies are first reduced to the exact bytes each token decodes to
// (Vocabulary). A Bridge then compiles, once per pair, the map phi that routes
// every teacher token to the longest student token whose bytes prefix the
// teacher token's bytes. At a position where both segmentations share a
// boundary, the student target is the scatter of the teacher mass through phi
// (the paper's Eq. 4); inside a teacher token, it is the same scatter over the
// teacher tokens that continue the bytes already emitted, normalized by their
// mass (Eq. 5). Both are the exact byte-prefix marginal while the student
// token does not cross a teacher boundary. When it does, the target receives
// the realized-path chain value (Eq. 6), a lower bound, moved from the head
// cell so that the total mass is unchanged.
//
// Every target accounts for all of the teacher's mass: student cells, the
// residual of byte-bearing teacher tokens no student token prefixes, the mass
// on declared stop tokens, the mass on byte-less teacher tokens, and the mass
// outside the teacher's top-k. None of it is renormalized away.
//
// The package reads tokenizer metadata and computes targets. It does not load
// a model, run a forward pass, or write a cache.
package bridge
