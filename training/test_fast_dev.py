"""
Test the Fast/Dev Recommender
Quick demo after training
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.infer import MovieRecommender

def main():
    print("=" * 80)
    print("🎬 Testing Fast/Dev Recommender")
    print("=" * 80)
    print()
    
    # Check if model exists
    model_dir = Path(__file__).parent / "models_fast"
    
    if not model_dir.exists():
        print(f"❌ Model not found at {model_dir}")
        print("Please run train_fast_dev.py first to train the model.")
        return
    
    print(f"✅ Loading model from: {model_dir}")
    print()
    
    # Load the recommender
    recommender = MovieRecommender(str(model_dir))
    print()
    
    # Test movies to search for
    test_movies = ["Inception", "The Matrix", "Interstellar", "Dark Knight"]
    
    print("=" * 80)
    print("🔍 Testing Recommendations")
    print("=" * 80)
    print()
    
    for movie_title in test_movies:
        print(f"🎬 Looking for recommendations for: {movie_title}")
        
        # Find the movie
        matched_title = recommender.find_movie(movie_title)
        
        if not matched_title:
            print(f"  ❌ Movie not found in dataset")
            
            # Try searching
            search_results = recommender.search_movies(movie_title, n=5)
            if search_results:
                print(f"  💡 Did you mean:")
                for result in search_results:
                    print(f"     - {result}")
        else:
            print(f"  ✅ Found: {matched_title}")
            
            # Get recommendations
            try:
                results = recommender.get_recommendations(movie_title, n=5)
                
                if 'error' in results:
                    print(f"  ❌ {results['error']}")
                else:
                    print(f"  📊 Top 5 Recommendations:")
                    for i, rec in enumerate(results.get('recommendations', []), 1):
                        title = rec.get('title', 'N/A')
                        rating = rec.get('rating', 'N/A')
                        similarity = rec.get('similarity_score', 0)
                        print(f"    {i}. {title} (⭐ {rating}/10, Similarity: {similarity:.3f})")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    main()
