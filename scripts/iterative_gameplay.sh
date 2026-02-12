#!/bin/bash

# Configuration
INITIAL_GAMES=100
GAMES_PER_ITER=50
EPOCHS=100
BATCH_SIZE=128
MAX_ITERATIONS=10
EVAL_GAMES=50  # Games to play for evaluation after each iteration
EARLY_STOPPING=10  # Stop training if validation loss doesn't improve for N epochs (0 = disabled)

# Temperature scheduling: start high (exploration), decrease over time (exploitation)
INITIAL_TEMP=1.5
FINAL_TEMP=0.5

# Directories
DATA_DIR="trainingdata"
MODEL_DIR="models"
EVAL_DIR="evaluations"

mkdir -p ${EVAL_DIR}

# Start with random agent
CURRENT_MODEL="random"
ITERATION=1

# First iteration: generate with random agent
echo "=== Iteration $ITERATION: Generating $INITIAL_GAMES games with random agent ==="
time .build/release/orion generate -o ${DATA_DIR}/data_r1_${INITIAL_GAMES} -g ${INITIAL_GAMES} -a ${CURRENT_MODEL} -t ${INITIAL_TEMP}

echo "=== Iteration $ITERATION: Training model ==="
CURRENT_MODEL="${MODEL_DIR}/model_g${ITERATION}_e${EPOCHS}_b${BATCH_SIZE}"
time .build/release/orion train -i ${DATA_DIR}/data_r1_${INITIAL_GAMES}.gz -e ${EPOCHS} -b ${BATCH_SIZE} -o ${CURRENT_MODEL} --early-stopping ${EARLY_STOPPING}

# Evaluate against random
echo "=== Iteration $ITERATION: Evaluating model vs random ==="
.build/release/orion play -n ${EVAL_GAMES} -a ${CURRENT_MODEL}/ random > ${EVAL_DIR}/eval_iter${ITERATION}.txt 2>&1
echo "Evaluation results saved to ${EVAL_DIR}/eval_iter${ITERATION}.txt"

# Accumulate training data
ACCUMULATED_DATA="${DATA_DIR}/accumulated_iter${ITERATION}.gz"
cp ${DATA_DIR}/data_r1_${INITIAL_GAMES}.gz ${ACCUMULATED_DATA}

# Subsequent iterations: generate with previous model, then train
for ITERATION in $(seq 2 ${MAX_ITERATIONS}); do
   echo ""
   echo "=== Iteration $ITERATION ==="

   # Calculate temperature (linear decay from INITIAL_TEMP to FINAL_TEMP)
   TEMP=$(echo "scale=2; ${INITIAL_TEMP} - (${INITIAL_TEMP} - ${FINAL_TEMP}) * ($ITERATION - 1) / (${MAX_ITERATIONS} - 1)" | bc)
   echo "Using temperature: ${TEMP}"

   # Generate new data with current model
   echo "Generating $GAMES_PER_ITER games with model from iteration $((ITERATION-1))..."
   time .build/release/orion generate -o ${DATA_DIR}/data_g$((ITERATION-1))_${GAMES_PER_ITER} -g ${GAMES_PER_ITER} -a ${CURRENT_MODEL}/ -t ${TEMP}

   # Combine with accumulated data (if you want to accumulate, uncomment these lines)
   # echo "Combining with previous training data..."
   # # Note: This requires loading both datasets and merging them
   # # For now, we'll just use the latest data

   # Train on latest data, starting from previous model
   echo "Training model (continuing from previous iteration)..."
   PREV_MODEL="${MODEL_DIR}/model_g$((ITERATION-1))_e${EPOCHS}_b${BATCH_SIZE}"
   CURRENT_MODEL="${MODEL_DIR}/model_g${ITERATION}_e${EPOCHS}_b${BATCH_SIZE}"
   time .build/release/orion train -i ${DATA_DIR}/data_g$((ITERATION-1))_${GAMES_PER_ITER}.gz -e ${EPOCHS} -b ${BATCH_SIZE} -m ${PREV_MODEL}/ -o ${CURRENT_MODEL} --early-stopping ${EARLY_STOPPING}

   # Evaluate against random
   echo "Evaluating model vs random..."
   .build/release/orion play -n ${EVAL_GAMES} -a ${CURRENT_MODEL}/ random > ${EVAL_DIR}/eval_iter${ITERATION}.txt 2>&1
   echo "Evaluation results saved to ${EVAL_DIR}/eval_iter${ITERATION}.txt"

   # Also evaluate against previous model
   if [ -d "${PREV_MODEL}" ]; then
      echo "Evaluating model vs previous iteration..."
      .build/release/orion play -n ${EVAL_GAMES} -a ${CURRENT_MODEL}/ ${PREV_MODEL}/ > ${EVAL_DIR}/eval_iter${ITERATION}_vs_prev.txt 2>&1
   fi
done

echo ""
echo "=== Training complete! Final model: ${CURRENT_MODEL} ==="
echo "Evaluation results in: ${EVAL_DIR}/"
