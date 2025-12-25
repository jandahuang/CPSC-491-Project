import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import MovieRecommenderTrainer

def main():
    print("=" * 80)
    print("🚀 FAST/DEV Configuration - 10K Movies Training")
    print("=" * 80)
    print()
    
    dataset_path = Path(__file__).parent.parent / "TMDB_movie_dataset_v11.csv"
    
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        print()
        print("To download the TMDB dataset, run:")
        print()
        print("  pip install kagglehub")
        print("  python -c \"import kagglehub; path = kagglehub.dataset_download('asaniczka/tmdb-movies-dataset-2023-930k-movies'); print(path)\"")
        print()
        print("Then extract TMDB_movie_dataset_v11.csv to the project root directory.")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    print()
    
    trainer = MovieRecommenderTrainer(
        output_dir='./models_fast',
        use_dimensionality_reduction=False  # Disable SVD for speed
    )
    
    print("📊 Training Configuration:")
    print("  - Max movies: 10,000")
    print("  - Quality threshold: high (500+ votes)")
    print("  - SVD reduction: disabled (faster training)")
    print("  - Expected time: ~2 minutes")
    print("  - Expected RAM: ~500MB")
    print("  - Expected model size: ~40MB")
    print()
    
    print("🔄 Starting training...")
    print()
    
    df, sim_matrix = trainer.train(
        str(dataset_path),
        quality_threshold='high',  # Only 500+ votes
        max_movies=10000
    )
    
    print()
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Model saved to: ./models_fast")
    print(f"Movies trained: {len(df):,}")
    print(f"Model files created:")
    print(f"  - movie_metadata.parquet")
    print(f"  - similarity_matrix.npy")
    print(f"  - title_to_idx.json")
    print(f"  - config.json")
    print()
    print("Next: Load and test the recommender...")
    print()

if __name__ == "__main__":
    main()
