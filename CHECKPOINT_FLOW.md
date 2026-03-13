# Checkpoint Saving Flow Chart

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING INITIALIZATION                       │
├─────────────────────────────────────────────────────────────────┤
│ Initialize tracking variables:                                  │
│   • bestValidationLoss = ∞                                      │
│   • epochsWithoutImprovement = 0                                │
│   • bestModelEpoch = 0                                          │
│   • epochLosses = []                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  FOR EACH EPOCH │
                    │  (1 to epochs)  │
                    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  1. TRAIN ON TRAINING SET           │
        │     • Process batches                │
        │     • Compute avgEpochLoss          │
        │     • Append to epochLosses[]         │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  2. VALIDATE ON VALIDATION SET       │
        │     • Process validation batches    │
        │     • Compute avgValidationLoss      │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  3. PRINT EPOCH STATS               │
        │     Train Loss, Val Loss, etc.      │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  4. CHECK IF NEW BEST MODEL          │
        │     avgValidationLoss <              │
        │     bestValidationLoss?              │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                 NO
                    │                   │
                    ▼                   ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ NEW BEST FOUND!      │  │ NO IMPROVEMENT       │
    │                      │  │                      │
    │ • bestValidationLoss │  │ • epochsWithout      │
    │   = avgValidationLoss│  │   Improvement++       │
    │ • epochsWithout      │  │ • Print: "No         │
    │   Improvement = 0    │  │   improvement for    │
    │ • bestModelEpoch =   │  │   X/Y epochs"        │
    │   epoch              │  │                      │
    │                      │  └──────────────────────┘
    │ • Print improvement  │              │
    │                      │              ▼
    │ • IF earlyStopping   │  ┌──────────────────────┐
    │   > 0:               │  │ CHECK EARLY STOPPING  │
    │   SAVE BEST MODEL    │  │ epochsWithout >=      │
    │   to disk:           │  │ earlyStopping?         │
    │   {outputPath}_best  │  └──────────────────────┘
    │   (with metadata)    │              │
    └──────────────────────┘      ┌───────┴───────┐
                    │              │               │
                    │             YES              NO
                    │              │               │
                    │              ▼               │
                    │  ┌──────────────────────┐    │
                    │  │ BREAK TRAINING LOOP  │    │
                    │  │ Print early stop msg │    │
                    │  └──────────────────────┘    │
                    │              │               │
                    └──────────────┴──────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────┐
        │  5. CHECK REGULAR CHECKPOINT        │
        │     epoch % saveInterval == 0       │
        │     OR epoch == epochs?             │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                 NO
                    │                   │
                    ▼                   │
    ┌──────────────────────┐            │
    │ SAVE CHECKPOINT      │            │
    │ to disk:             │            │
    │ {outputPath} or      │            │
    │ models/checkpoint_   │            │
    │   epoch{epoch}.mlx   │            │
    │                      │            │
    │ (with metadata       │            │
    │  including:          │            │
    │  - trainingEpochs   │            │
    │  - trainingLoss =    │            │
    │    avgValidationLoss)│            │
    └──────────────────────┘            │
                    │                   │
                    └───────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  NEXT EPOCH?    │
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                 NO
                    │                   │
                    └───────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  6. SAVE FINAL MODEL                 │
        │     (if outputPath provided)         │
        │                                      │
        │     Save to: {outputPath}            │
        │     (with metadata:                  │
        │      - trainingEpochs =              │
        │        bestModelEpoch if early       │
        │        stopped, else epochs          │
        │      - trainingLoss =                │
        │        bestValidationLoss)           │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  7. PRINT SUMMARY                    │
        │     • Final model saved              │
        │     • Best validation loss           │
        │     • Best epoch                     │
        │     • If early stopping:             │
        │       "Best model at {path}_best"    │
        └─────────────────────────────────────┘
```

## Key Points

### Variables Tracked in Memory:
- **`bestValidationLoss`**: Best validation loss seen so far (initialized to ∞)
- **`bestModelEpoch`**: Epoch number where best validation loss occurred
- **`epochsWithoutImprovement`**: Counter for early stopping
- **`epochLosses[]`**: Array of training losses per epoch (for plotting)

### When Models Are Saved to Disk:

1. **Best Model Checkpoint** (`{outputPath}_best`):
   - **When**: Every time validation loss improves (if `earlyStoppingPatience > 0`)
   - **Contains**: Model weights + metadata with best validation loss
   - **Purpose**: Preserve the best model for generalization

2. **Regular Checkpoints** (`{outputPath}` or `models/checkpoint_epoch{N}.mlx`):
   - **When**: Every `saveInterval` epochs (default: 5) OR at final epoch
   - **Contains**: Model weights + metadata with current validation loss
   - **Purpose**: Periodic snapshots for recovery/resume

3. **Final Model** (`{outputPath}`):
   - **When**: After all epochs complete (or early stopping)
   - **Contains**: Final model weights + metadata
   - **Note**: This is the LAST epoch's model, not necessarily the best!

### Important Notes:

⚠️ **The final model is NOT necessarily the best model!**
- If early stopping is enabled, the best model is saved separately as `{outputPath}_best`
- The final model at `{outputPath}` is the model from the last epoch
- **Always use `{outputPath}_best` for best generalization** if early stopping was used

### Metadata Saved with Each Checkpoint:
- `version`: Model version string
- `architectureVersion`: Architecture version number
- `createdAt`: Creation timestamp
- `trainingEpochs`: Epoch number when saved
- `trainingLoss`: Validation loss at that epoch
- `description`: Optional description
- `checksum`: Optional checksum

### Early Stopping Logic:
- If `earlyStoppingPatience = 0`: Early stopping disabled
- If `earlyStoppingPatience > 0`: Training stops when validation loss doesn't improve for N consecutive epochs
- Best model is automatically saved when found (if early stopping enabled)

