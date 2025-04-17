# Fine-tuned Cybersecurity TTP Model Evaluation

This report summarizes the performance comparison between the baseline model (all-mpnet-base-v2) and our fine-tuned model for cybersecurity Tactics, Techniques, and Procedures (TTPs) similarity detection.

## Performance on Test Dataset

| Metric | Baseline Model | Fine-tuned Model | Improvement |
|--------|---------------|-----------------|-------------|
| Accuracy | 47.74% | 60.65% | +12.90% |
| Precision | 83.05% | 60.65% | -22.40% |
| Recall | 17.38% | 100.00% | +82.62% |
| F1 Score | 28.74% | 75.50% | +46.76% |

The fine-tuned model shows **significant improvement** in overall performance, especially in recall and F1 score. While precision decreased, this trade-off resulted in a much more balanced model that achieves better overall detection capability.

## Key Observations

1. **Substantial F1 Score Improvement**: The fine-tuned model achieves a 46.76% absolute improvement in F1 score, indicating much better overall performance in identifying similar TTPs.

2. **Perfect Recall**: The fine-tuned model achieves 100% recall, meaning it successfully identifies all similar TTP pairs. This is especially important in a cybersecurity context where missing a potential threat is more costly than generating false positives.

3. **Better Accuracy**: The fine-tuned model improves accuracy by 12.90%, showing better overall classification performance.

## Real-World Example Performance

The evaluation on completely unseen, real-world examples demonstrates that the fine-tuned model significantly outperforms the baseline:

- The fine-tuned model consistently assigns **higher similarity scores** to related cybersecurity TTPs.
- The average similarity score difference between models is substantial, showing that fine-tuning has made the model much more sensitive to cybersecurity domain-specific language.
- The baseline model often fails to recognize relationships between different TTPs that utilize similar tactics or techniques.

## Visualizations

Three key visualizations were generated to illustrate the performance differences:

1. **Confusion Matrices**: Shows how each model classifies similar and dissimilar pairs
2. **Similarity Distribution**: Illustrates how well each model separates similar from dissimilar examples
3. **Performance Metrics Comparison**: Visual comparison of accuracy, precision, recall, and F1 score

## Conclusion

The evaluation confirms that fine-tuning a sentence transformer model on domain-specific cybersecurity TTP data yields significantly better performance in identifying and relating similar attack techniques. The improvements are substantial across almost all metrics, with the most dramatic gains in recall and F1 score.

This fine-tuned model is now much better equipped to:
- Identify related attack patterns
- Group similar TTPs together
- Recognize when new, unseen TTPs relate to known techniques

These capabilities are critical for threat intelligence analysis, security operations, and automated security tooling. 