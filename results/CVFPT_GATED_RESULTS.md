# CVFPT Context Vector Comparison Experiment

## Objective

Compare context vectors obtained through two different methods:

1. **Fixed-Point Contexts**: Context vectors after repetitive training (repeating each token 10 times)
2. **Single-Pass Contexts**: Context vectors from a single forward pass (no repetition)

## Hypothesis

If CVFPT (Context Vector Fixed Point Training) is effective, we expect:

- **High similarity** between fixed-point and single-pass contexts
- **Fast convergence** to fixed points (few repetitions needed)
- **Consistent patterns** across different tokens

## Experimental Setup

- **Number of tokens tested**: 100
- **Repetitions for fixed-point**: 10
- **Context vector dimension**: 256

## Results Summary

### Key Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| L2 Distance | 3.6943 | 0.1105 | 3.4136 | 4.0317 |
| Cosine Similarity | 0.9735 | 0.0016 | 0.9685 | 0.9774 |
| Correlation | 0.9735 | 0.0016 | 0.9685 | 0.9774 |
| Convergence Steps | 9.0000 | 0.0000 | 9.0000 | 9.0000 |
| Fixed-Point Norm | 16.0548 | 0.0006 | 16.0534 | 16.0561 |
| Single-Pass Norm | 16.0501 | 0.0007 | 16.0472 | 16.0515 |

### Interpretation

**L2 Distance** (Average: 3.694)

❌ **High** - Significant difference between the two methods

**Cosine Similarity** (Average: 0.973)

✅ **Very High** - Context vectors point in nearly the same direction

**Convergence Steps** (Average: 9.0/10)

❌ **Slow Convergence** - Many tokens do not reach fixed points

## Sample Tokens Analysis

Top 10 tokens with highest similarity:

| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |
|------|-------|---------|---------|------|------------|
| 1 | `ledge` | 3.414 | 0.977 | 0.977 | 9 |
| 2 | ` regul` | 3.463 | 0.977 | 0.977 | 9 |
| 3 | ` experience` | 3.490 | 0.976 | 0.976 | 9 |
| 4 | ` way` | 3.522 | 0.976 | 0.976 | 9 |
| 5 | `BC` | 3.536 | 0.976 | 0.976 | 9 |
| 6 | ` increased` | 3.537 | 0.976 | 0.976 | 9 |
| 7 | `pos` | 3.552 | 0.976 | 0.976 | 9 |
| 8 | `aches` | 3.566 | 0.975 | 0.975 | 9 |
| 9 | `>>` | 3.569 | 0.975 | 0.975 | 9 |
| 10 | ` asked` | 3.570 | 0.975 | 0.975 | 9 |


Top 10 tokens with lowest similarity:

| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |
|------|-------|---------|---------|------|------------|
| 1 | ` harm` | 4.032 | 0.968 | 0.968 | 9 |
| 2 | ` date` | 3.954 | 0.970 | 0.970 | 9 |
| 3 | ` quality` | 3.921 | 0.970 | 0.970 | 9 |
| 4 | ` pan` | 3.914 | 0.970 | 0.970 | 9 |
| 5 | ` us` | 3.911 | 0.970 | 0.970 | 9 |
| 6 | ` patients` | 3.893 | 0.971 | 0.971 | 9 |
| 7 | `ctions` | 3.883 | 0.971 | 0.971 | 9 |
| 8 | ` operation` | 3.874 | 0.971 | 0.971 | 9 |
| 9 | ` +` | 3.862 | 0.971 | 0.971 | 9 |
| 10 | `Add` | 3.858 | 0.971 | 0.971 | 9 |

## Conclusions

⚠️ **Mixed Results**

The metrics suggest that:

- Some tokens converge to stable fixed points
- Other tokens show significant variation
- Further training or architectural changes may be needed

## Visualizations

See `cvfpt_comparison.png` for detailed plots.

## Raw Data

Full experimental data saved to `cvfpt_comparison_data.npz`
