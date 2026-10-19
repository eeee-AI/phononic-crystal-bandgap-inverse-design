# Pre-publication checklist

Use this checklist before creating the public repository release.

- [ ] Replace `[REPOSITORY_URL]` in `DATA_AVAILABILITY.md` and `CODE_AVAILABILITY.md`.
- [ ] Confirm the final author list, corresponding-author email, manuscript title, and journal data policy.
- [ ] Do not add any full-data CSV, `128x128` source-image directory, bulk parameter mapping, model checkpoint, Abaqus `.odb`, batch outputs, or propagation video.
- [ ] Keep the existing `.gitignore` in place before staging files.
- [ ] Create a **public, empty GitHub repository** named `phononic-crystal-minimal-reproduction` under the research account. Do not initialise it with a README, licence, or `.gitignore`.
- [ ] Push only this package, then create a tagged release such as `v1.0.0` after the manuscript is accepted or when the journal asks for a permanent revision.
- [ ] Archive the tagged release with Zenodo or the journal-approved repository if a DOI is required.

The full dataset should remain in a separate private archive. Never add it to the public repository and delete it later: Git history would still preserve it.
