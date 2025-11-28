#!/bin/bash
#SBATCH --job-name=market_transformer
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load necessary modules (adjust versions as needed based on Zaratan availability)
module load python/3.10
module load cuda/11.8

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the training script
echo "Starting training..."
python scripts/train.py --epochs 100 --batch_size 32 --gpu

echo "Training complete."
