**TMDB Movies Dataset 2023** - 1.3M+ movies with rich metadata

### Key Features:
- ✅ **Single CSV file** (no merging needed)
- ✅ **1,332,407 movies** total
- ✅ **Rich metadata**: genres, keywords, production companies, countries
- ✅ **Quality metrics**: vote_average, vote_count, popularity
- ✅ **IMDB integration**: imdb_id for cross-referencing
- ✅ **Poster images**: Direct URLs to movie posters

### Dataset Columns Used:
```
✅ title               - Movie title
✅ genres              - Multiple genres per movie
✅ keywords            - Descriptive keywords
✅ production_companies - Production studios
✅ production_countries - Countries of production
✅ overview            - Plot summary (up to 50 words)
✅ tagline             - Movie tagline
✅ vote_average        - Rating (0-10)
✅ vote_count          - Number of votes
✅ popularity          - TMDB popularity score
✅ release_date        - Release date
✅ imdb_id             - IMDB identifier
✅ poster_path         - Poster image path
```


**Three quality tiers:**
```python
# Low Quality: 5+ votes (maximum dataset size)
trainer.train(quality_threshold='low')      # ~930K movies

# Medium Quality: 50+ votes (recommended - balanced)
trainer.train(quality_threshold='medium')   # ~200K movies

# High Quality: 500+ votes (highest quality)
trainer.train(quality_threshold='high')     # ~50K movies
```

### **Advanced Filtering**

```python
# Filter by year range
recommender.get_recommendations(
    "Inception",
    min_year=2015,
    max_year=2023
)

# Filter by rating
recommender.get_recommendations(
    "The Matrix",
    min_rating=7.5  # Only highly rated
)

# Filter by genres (multiple)
recommender.get_recommendations(
    "Interstellar",
    genres=['Science Fiction', 'Drama']
)

# Exclude same production company
recommender.get_recommendations(
    "Avatar",
    exclude_same_company=True
)

# Combine all filters
recommender.get_recommendations(
    "The Dark Knight",
    n_recommendations=10,
    min_year=2010,
    max_year=2023,
    genres=['Action', 'Thriller'],
    min_rating=7.0,
    exclude_same_company=True
)
```

## 📊 Recommended Configurations

### Configuration 1: Full Dataset (930K+ movies)
**Requirements:** 16GB+ RAM, GPU recommended
```python
trainer = MovieRecommenderTrainer(
    output_dir='./models_full',
    use_dimensionality_reduction=True,
    n_components=400  # Lower for stability
)

df, sim = trainer.train(
    path,
    quality_threshold='low',  # 5+ votes
    max_movies=None  # All movies
)
```
- Training time: ~45-60 min
- Memory: ~4-6GB during training
- Model size: ~800MB
- Best for: Complete movie database

### Configuration 2: High Quality (100K movies) ⭐ RECOMMENDED
**Requirements:** 8GB RAM
```python
trainer = MovieRecommenderTrainer(
    output_dir='./models',
    use_dimensionality_reduction=True,
    n_components=500
)

df, sim = trainer.train(
    path,
    quality_threshold='medium',  # 50+ votes
    max_movies=100000  # Top 100K
)
```
- Training time: ~15 min
- Memory: ~2GB during training
- Model size: ~180MB
- Best for: Production deployment

### Configuration 3: Fast Training (10K movies)
**Requirements:** 4GB RAM
```python
trainer = MovieRecommenderTrainer(
    output_dir='./models_fast',
    use_dimensionality_reduction=False
)

df, sim = trainer.train(
    path,
    quality_threshold='high',  # 500+ votes
    max_movies=10000
)
```
- Training time: ~2 min
- Memory: ~500MB
- Model size: ~40MB
- Best for: Testing/development

## 🎯 Complete Usage Example

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy nltk kagglehub
```

### Step 2: Download Dataset & Train
```python
import kagglehub
from movie_recommender_trainer import MovieRecommenderTrainer

# Dataset path (already downloaded in project folder)
path = "./TMDB_movie_dataset_v11.csv"

# Train model (recommended config)
trainer = MovieRecommenderTrainer(
    output_dir='./models',
    use_dimensionality_reduction=True,
    n_components=500
)

df, sim_matrix = trainer.train(
    path,
    quality_threshold='medium',
    max_movies=100000
)
```

### Step 3: Load & Use Recommender
```python
from movie_recommender_inference import MovieRecommender

# Load trained model
recommender = MovieRecommender(model_dir='./models')

# Get recommendations
results = recommender.get_recommendations(
    "Inception",
    n_recommendations=10,
    min_rating=7.0
)

# Print results
recommender.print_recommendations(results, show_scores=True)
```
