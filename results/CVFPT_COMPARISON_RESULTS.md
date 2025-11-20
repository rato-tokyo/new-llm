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
| L2 Distance | 0.5875 | 0.1146 | 0.2420 | 0.8385 |
| Cosine Similarity | 0.9993 | 0.0003 | 0.9986 | 0.9999 |
| Correlation | 0.9993 | 0.0003 | 0.9986 | 0.9999 |
| Convergence Steps | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fixed-Point Norm | 15.8872 | 0.0000 | 15.8872 | 15.8872 |
| Single-Pass Norm | 15.8896 | 0.0005 | 15.8882 | 15.8906 |

### Interpretation

**L2 Distance** (Average: 0.588)

✅ **Very Low** - Fixed-point and single-pass contexts are nearly identical

**Cosine Similarity** (Average: 0.999)

✅ **Very High** - Context vectors point in nearly the same direction

**Convergence Steps** (Average: 1.0/10)

✅ **Fast Convergence** - Contexts reach fixed points quickly

## Sample Tokens Analysis

Top 10 tokens with highest similarity:

| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |
|------|-------|---------|---------|------|------------|
| 1 | `ously` | 0.242 | 1.000 | 1.000 | 1 |
| 2 | ` 30` | 0.378 | 1.000 | 1.000 | 1 |
| 3 | ` watch` | 0.390 | 1.000 | 1.000 | 1 |
| 4 | `verage` | 0.411 | 1.000 | 1.000 | 1 |
| 5 | ` something` | 0.418 | 1.000 | 1.000 | 1 |
| 6 | `ram` | 0.423 | 1.000 | 1.000 | 1 |
| 7 | `34` | 0.427 | 1.000 | 1.000 | 1 |
| 8 | ` October` | 0.436 | 1.000 | 1.000 | 1 |
| 9 | `isions` | 0.444 | 1.000 | 1.000 | 1 |
| 10 | ` UK` | 0.445 | 1.000 | 1.000 | 1 |


Top 10 tokens with lowest similarity:

| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |
|------|-------|---------|---------|------|------------|
| 1 | ` id` | 0.839 | 0.999 | 0.999 | 1 |
| 2 | ` my` | 0.828 | 0.999 | 0.999 | 1 |
| 3 | ` Ex` | 0.813 | 0.999 | 0.999 | 1 |
| 4 | ` from` | 0.791 | 0.999 | 0.999 | 1 |
| 5 | ` fail` | 0.776 | 0.999 | 0.999 | 1 |
| 6 | ` Sol` | 0.770 | 0.999 | 0.999 | 1 |
| 7 | `77` | 0.768 | 0.999 | 0.999 | 1 |
| 8 | ` look` | 0.757 | 0.999 | 0.999 | 1 |
| 9 | `irl` | 0.753 | 0.999 | 0.999 | 1 |
| 10 | `irl` | 0.753 | 0.999 | 0.999 | 1 |

## Conclusions

✅ **CVFPT is Effective**

The high cosine similarity and low L2 distance indicate that:

- The model successfully learns fixed-point representations
- Repetitive training converges to stable context vectors
- Single-pass contexts are good approximations of fixed points

## Visualizations

See `cvfpt_comparison.png` for detailed plots.

## Raw Data

Full experimental data saved to `cvfpt_comparison_data.npz`
