# Reference Implementation
This is a slow reference implementation of the algorithms
presented in the paper. The implementation closely mirrors
the pseudocode. You only need Numpy and Matplotlib to
run the reference.

`segment_tree.py` contains the code
to generate a segment tree and the STSample procedure.
`lemma_sampler` implements the construction and sampling procedure
of the lemma presented in the section "A Simpler Row Sampling
Problem". Finally, `krp_sampler.py` implements the procedures
"ConstructKRPSampler" and "KRPSample".

To test the implementation, run the following while inside
this directory:
```python
python test_reference_implementation.py
```
If all goes well, you will see a file 
`distribution_comparison_refence_impl` in the directory
`plotting` in the repository root. This compares the
distribution of true leverage scores against a histogram
of draws from the reference, which should match closely.

