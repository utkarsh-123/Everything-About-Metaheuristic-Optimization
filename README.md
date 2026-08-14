CEC 2022 Suite features 12 core functions (divided into Unimodal, Basic Multimodal, Hybrid, and Composition functions). 
It is highly popular because it removed structural redundancies present in older versions.

To conform to standard CEC benchmarking protocols when evaluating your metaheuristic algorithm, ensure your testing pipeline implements these parameters:
    
    Search Bounds: Fixed at (-100,100)^D for standard numeric functions.
    Dimensions (D): Test your algorithm across progressive scales, usually D = 10, 20, 30, 50, and 100.
    Max Function Evaluations (MaxFEs): Set strictly to 10,000 × D.
    Statistical Validation: Perform 30 independent runs per function. 
    Compare results using non-parametric checks like the Wilcoxon rank-sum test (α = 0.05) and the Friedman test for global ranking.
