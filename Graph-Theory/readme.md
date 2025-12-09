# Google PageRank Algorithm: Graph Theory Implementation

## Overview

This project demonstrates the implementation of Google's PageRank algorithm using NetworkX to compute page importance scores on a directed web graph. The assignment fulfills two primary objectives:

1. **Research & Theoretical Analysis**: Explain the PageRank algorithm using Graph Theory, Linear Algebra, and Probability Theory concepts
2. **Practical Implementation**: Compute PageRank scores for a directed graph of web pages using the NetworkX library

The repository includes a complete Jupyter notebook, Python implementations, and CSV output of results.

## Assignment Objectives

### Part 1: Theoretical Research
- Understand the Random Surfer Model and its relationship to Markov chains
- Analyze the transition (stochastic) matrix representation of web graphs
- Interpret PageRank as the principal eigenvector of the transition matrix
- Explain the damping factor's role in ensuring convergence
- Document the power-iteration method for PageRank computation

### Part 2: Practical Implementation
- Model the provided directed web graph using NetworkX
- Compute PageRank scores using the `nx.pagerank()` function
- Visualize results with node sizes proportional to PageRank values
- Export numerical results to CSV format

## Files

| File | Description |
|------|-------------|
| `PageRank_Notebook.ipynb` | Complete Jupyter notebook with full implementation, visualizations, and mathematical explanations |
| `page_rank_example.py` | Standalone Python script for computing and visualizing PageRank |
| `PageRank_Assignment.md` | Assignment requirements and reference documentation |
| `pagerank_results.csv` | CSV output with PageRank scores and percentages (generated after running) |
| `pagerank_visualization.png` | Network graph visualization with node sizes scaled by PageRank (generated after running) |
| `README.md` | This file |

## Quick Start

### Run the Jupyter Notebook
```bash
# Navigate to the Graph-Theory directory
cd E:\ML-Models\Graph-Theory

# Launch Jupyter (if installed)
jupyter notebook PageRank_Notebook.ipynb
```

### Run the Python Script
```bash
cd E:\ML-Models\Graph-Theory
python page_rank_example.py
```

**Output files created:**
- `pagerank_results.csv` — PageRank scores with percentages
- `pagerank_visualization.png` — Network graph visualization

## Graph Structure

**Nodes**: A, B, C, D, E, F, g1, g2, g3, g4, g5 (11 total)

**Edges**: 19 directed hyperlinks representing web page connections (A→B, D→B, D→A, etc.)

The graph models a hypothetical web structure where nodes are web pages and directed edges represent hyperlinks between pages.

## Mathematical Foundation

### Key Concepts

1. **Random Surfer Model**: Models web browsing as a random walk where a surfer:
   - Follows outgoing links with probability α (damping factor, typically 0.85)
   - Jumps to any random page with probability (1-α) to handle dangling nodes

2. **Transition Matrix**: A stochastic matrix M where M[i,j] = probability of moving from page i to page j

3. **Eigenvector Interpretation**: PageRank is the principal eigenvector of the modified transition matrix, representing steady-state visit probability

4. **Power Iteration**: Efficient iterative algorithm for computing PageRank:
   ```
   r(t+1) = α * M * r(t) + (1-α) * e / n
   ```
   Converges when ||r(t+1) - r(t)|| < tolerance

### Parameters

- **Damping Factor (α)**: 0.85 (default) — adjustable in code
- **Tolerance**: 1.0e-06 — convergence threshold
- **Max Iterations**: None (power iteration continues until tolerance met)

## Results Interpretation

PageRank scores represent the relative importance/authority of each page:
- Higher scores indicate pages that are more frequently "visited" by the random surfer
- Scores are normalized to sum to 1.0
- Percentages show each page's share of total PageRank

Example output:
```
Node    PageRank Score    Percentage
B       0.387927          38.79%
C       0.363331          36.33%
E       0.093933          9.39%
...
```

## Customization

### Modify the Graph Topology
Edit the `edges` list in the notebook or Python script:
```python
edges = [
    ("A", "B"),  # A links to B
    ("D", "B"),  # D links to B
    # Add or modify edges here
]
```

### Adjust Algorithm Parameters
```python
# Change damping factor
pr = compute_pagerank(G, alpha=0.90, tol=1.0e-06)
```

## Requirements

**Python Libraries:**
- `networkx` — Graph algorithms
- `matplotlib` — Visualization
- `csv` — Built-in module for data export

**Installation:**
```bash
pip install networkx matplotlib
```

## Submission Format

As per assignment requirements, submit as a **PDF with an active link to a Google Colaboratory notebook**.

1. Create a copy of the Jupyter notebook in Google Colab
2. Run all cells to generate results
3. Export the notebook as PDF
4. Include active hyperlink to the Colab notebook in the PDF
5. Attach the PDF to your submission

## References

- **NetworkX Documentation**: https://networkx.org/documentation/stable/
- **PageRank Paper**: Lawrence et al., "The PageRank Citation Ranking: Bringing Order to the Web" (1998)
- **Graph Theory & Linear Algebra**: See `PageRank_Assignment.md` for detailed mathematical explanations

## Notes

- The damping factor (0.85) balances between following links and random jumps
- NetworkX's implementation uses sparse matrix operations for efficiency
- Results are deterministic given fixed graph topology and parameters
- Node labeling conventions: capital letters (A-F) for main nodes, lowercase g# for leaf nodes

