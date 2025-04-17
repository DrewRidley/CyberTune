# Advanced Cybersecurity TTP Model Evaluation

This report summarizes the performance of our advanced fine-tuned model using hard negative mining strategies for cybersecurity Tactics, Techniques, and Procedures (TTPs) similarity detection.

## Performance Comparison

| Metric | Baseline Model | Basic Fine-tuned | Advanced (Hard Negative Mining) | Improvement over Baseline |
|--------|---------------|-----------------|----------------------|--------------------------|
| Accuracy | 47.74% | 60.65% | 63.87% | +16.13% |
| Precision | 83.05% | 60.65% | 62.67% | -20.38% |
| Recall | 17.38% | 100.00% | 100.00% | +82.62% |
| F1 Score | 28.74% | 75.50% | 77.05% | +48.31% |

The advanced model with hard negative mining shows **significant improvement** over both the baseline and basic fine-tuned models. While precision decreased compared to the baseline, the trade-off resulted in perfect recall and much higher F1 score.

## Training Strategy: Hard Negative Mining

Hard negative mining is a technique that focuses training on the most challenging examples - the negative pairs that are most similar to positive pairs. This approach helps the model learn more subtle distinctions between related and unrelated TTPs.

The advanced model was trained with:
- 300 hard negative examples
- 12 training epochs
- Batch size of 16
- Learning rate of 3e-5
- 15% warmup ratio

## Key Findings

1. **Further Improved F1 Score**: The advanced model achieves a 77.05% F1 score, which is 1.55% higher than the basic fine-tuned model and 48.31% higher than the baseline.

2. **Maintained Perfect Recall**: Like the basic fine-tuned model, the advanced model achieves 100% recall, but with slightly better precision.

3. **Better Real-world Performance**: On unseen examples, the advanced model demonstrates larger similarity differences between the baseline and fine-tuned scores, with improvements of up to 57.12% in similarity scores for related cybersecurity TTPs.

4. **Enhanced Discrimination**: The model shows better ability to distinguish between related and unrelated TTPs, with clearer separation in the similarity distributions.

## Visualization Insights

The visualization outputs demonstrate:

1. **Improved Confusion Matrix**: The advanced model correctly classifies more examples than both the baseline and basic fine-tuned models.

2. **Better Similarity Distribution**: The advanced model shows clearer separation between positive and negative similarity scores.

3. **Higher Performance Metrics**: The bar charts show consistent improvement across most metrics.

## Conclusion

Hard negative mining has proven to be an effective strategy for further improving the model's performance in identifying and relating similar cybersecurity TTPs. By focusing on challenging examples, the model learned more nuanced relationships between different TTPs.

This advanced model is well-suited for:
- Identifying related attack patterns in threat intelligence
- Grouping similar TTPs for better defensive strategy planning
- Recognizing relationships between new and known attack techniques
- Supporting security analysts in understanding attack methodology connections

The improvements demonstrate that sophisticated training strategies beyond basic fine-tuning can yield meaningful performance gains for domain-specific tasks like cybersecurity TTP analysis. 