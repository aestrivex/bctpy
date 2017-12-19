# BCT 0.4.1

- Refactor code into multiple files
- Fix bug in efficiency_bin
- Fix bugs in modularity_louvain_und
- Fix bugs in `participation_coef_b*`
- Add some test cases

# BCT 0.4.0

- Add various new functions from Jan 2015 release of BCT
- Fix various bugs documented in github issues

# BCT 0.3.3

- Fix small bug in `latmio_und_connected` causing failure for sparse matrices
- Add non-networkx dependent algorithm to get_components (but less efficient)
- Add an implementation of consensus clustering and fix bug in agreement
- Fix bug causing `clustering_coef_bu` to always return 0
- Remembered to update changelog
- Fix some bugs in `modularity_louvain_dir` and related
- Fix bug in NBS and add optional paired-sample test statistic (sviter)

# BCT 0.3.2

- Change several functions including threshold_proportional and binarize have `copy=True` as default argument
- Fix bug in `threshold_proportional` where copying behavior did not work symmetric matrices.
- Fix minor quirk in `threshold_proportional` where `np.round` rounds to nearest even number (optimizes floating point) which is discrepant with BCT
- Add a test suite with some functions
- Fix typo in `rich_club_bu`
- Refactor `x[range(n),range(n)]` to `np.fill_diagonal`
- Fix off-by-one bug in `moduality_[prob/fine]tune_und_sign`

# BCT 0.3.1

- Fix bug in NBS
- Fix series of bugs in `null_models`

# BCT 0.3

- Added NBS
- Added in all of the new functions from the Dec 2013 release of BCT
- Fixed numerous bugs having to do with indexing errors in modularity
- Fixed several odd bugs with `clustering_coef`, `efficiency`, `distance`

For the next release, I clearly need a real test suite.
