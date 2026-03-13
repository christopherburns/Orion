# Training Tips for Self-Play

## Critical Issues Fixed

1. **Loading Previous Model**: The training command now uses `-m` to load the previous iteration's model, so training continues rather than starting from scratch.

2. **Temperature Scheduling**: Temperature starts high (1.5) for exploration and decreases to 0.5 for exploitation as the model improves.

3. **Evaluation**: Added evaluation matches after each iteration to track progress.

## Additional Recommendations

### 1. Lower Learning Rate for Fine-Tuning
After the first iteration, use a lower learning rate (e.g., 0.0001 instead of 0.001) since you're fine-tuning an existing model:

```bash
# First iteration: full learning rate
LR=0.001

# Subsequent iterations: lower learning rate
LR=0.0001
```

### 2. Accumulate Training Data
Instead of only using the latest iteration's data, accumulate data across iterations. This gives the model more diverse examples to learn from.

### 3. Check Loss Values
Monitor the training loss - if it's not decreasing, the model isn't learning:
- Policy loss should decrease from ~1.5-2.0 to ~0.8-1.2
- Value loss should decrease from ~0.5-1.0 to ~0.1-0.3

### 4. Use Greedy Play for Evaluation
When evaluating, use temperature=0 (greedy) to see the model's best play, not its exploratory play.

### 5. Start with More Initial Games
1000 games is good, but consider 2000-5000 for the first iteration to get better initial data.

### 6. Reduce Epochs Per Iteration
100 epochs might be too many - try 20-50 epochs per iteration. Too many epochs can cause overfitting to the current iteration's data.

### 7. Check Model Predictions
Verify the model is actually making reasonable predictions:
- Value predictions should be in [-1, 1] range
- Policy should have higher probabilities for legal moves
- Check if the model's move selection makes sense

### 8. Consider Curriculum Learning
Start with simpler scenarios or shorter games, then gradually increase complexity.

### 9. Monitor Win Rate
Track win rate against random over iterations - it should gradually increase.

### 10. If Still Not Working
- Try a larger model (more hidden units)
- Use a different optimizer (SGD with momentum)
- Check for bugs in move selection during generation
- Verify the loss function is correct
- Make sure the model is actually being saved/loaded correctly

