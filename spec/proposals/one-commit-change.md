# Proposal: change at one commit, with the equivalence check as the gate

Status: open. Source: the survey of 24 October 2026, `openie-research-experiments/rce/surveys/2026-10-24/continuous-improvement-without-versions.md`, read against 2.7, C.15, decision 64, F.4.3, F.5.3, 6.1 (T7), T3 and 6.10.

## What the record says

Dependency resolution with one version per package and arbitrary constraints is NP-complete (Mancinelli et al. 2006; Abate et al. 2020; Cox 2016); a semantic version is "a lossy summary of the perceived risk in adopting any particular change" (Winters, Manshreck and Wright 2020) and is violated in about a third of historical Maven releases (Raemaekers et al. 2017) and one in thirty top crates.io releases (Gruevski 2023); exposure is transitive (Decan, Mens and Grosjean 2019; Wetter and Ringland 2021). The record converges on identity from content (Dolstra 2006; Unison; Di Cosmo, Gruenpeter and Zacchiroli 2020), compatibility as a relation checked between two artifacts (Elm; Rust RFC 1105; Liskov and Wing 1994), additive change under names never recycled (Hickey 2016), coexistence instead of replacement, and a direct proof that a change preserved behaviour (Godlin and Strichman 2009; Lahiri et al. 2012). Hardware practice adds equivalence checking as the gate on every revision (Baumgartner et al. 2006) and audited baselines with problem-report-driven change control (ISO 10007; DO-254; ISO 26262-8).

The specification already states one language at one commit (2.7, C.15), definition hashes with a closed compatibility table (F.4.3), a manifest with no resolver (F.5.3), name-free log hashes and replay (T7, T3) and behaviour as the only sense of "same" (6.10). This proposal closes the gaps the survey lists.

## Rules proposed

1. **CC-1 Program hash.** A program (the set of modules given together, 4.1) has one hash: the SHA-256 of the canonical JSON of the list of its definition hashes (F.4.3) in canonical order, with `impl` blocks, `step`s and contracts included in each definition hash as F.4.3 already requires. It is name-free (T7) and layout-free. `rce hash` prints it; the Form JSON carries it; the genesis entry of D records it, so a log names the program that wrote it (Appendix E proposal C-3 supplies the version tag of the hash choice in the same entry).
2. **CC-2 The gate on a change.** A change to a program is classified by two checks run together: `rce api-diff` over every changed definition (F.4.3), and behaviour equality on the explored alphabet (6.10) with the program's examples replayed as regression vectors (6.9). A change to the text under which every accepted program's behaviour is unchanged is an erratum in the sense of C.15, or an extension when it also admits programs whose form the text did not have before (C.15's class for the Form gaining a form); neither needs a new release. Any other change to the text is a meaning change and a new release of the whole text. The decision of C.15 is thereby mechanical.
3. **CC-3 No asserted numbers.** No artifact of the language carries a MAJOR.MINOR.PATCH or any other asserted compatibility number. Appendix D's `spec NAME VERSION`, W-HS-VER-1, W-HS-VER-2 and "the version of δ at its node" in a RESOLVE entry are replaced by the definition hash of the handler spec and of δ and the `rce api-diff` verdict between the old and the new hash.
4. **CC-4 Provenance in emitted code.** An emitted file (10.5) begins with a comment naming the program hash (CC-1) and the commit of the checker that produced it, so an artifact is tied to its source as reproducible builds require (Lamb and Zacchiroli 2022).
5. **CC-5 Names.** A name is never reused for a different meaning (C.15, Hickey 2016); a removal is a new release (CONTRIBUTING rule 3); the one-time standard library renames of decision 64 stand, recorded.
6. **CC-6 The release label.** The `# rce-version:` line is retired from the text: it already selects nothing (2.7). A module that still carries it is read as before (the line is a comment). The manifest's `language` key names the specification commit, not a label.

## What changes in the normative text

2.7 (the label line retired), 6.1 (genesis records the program hash), 10.5 (provenance comment), 10.8 (`rce hash` prints the program hash), Appendix D (D.3, D.10: hashes replace versions), F.4.3 (the hash covers `impl`, `step` and contracts in the checker as the text requires), F.5.3 (`language` key), C.15 (the erratum rule stated mechanically by CC-2).

## What stays open

Proofs of T50 (equal hash, equal meaning) and T54 (manifest); the Lean statement of CC-2's erratum criterion as a theorem over 6.10; whether the explored alphabet of 6.10 under `--fuel` is enough evidence for CC-2 or the examples must cover every declared event (T59 bounds the evidence by 2^-n).
