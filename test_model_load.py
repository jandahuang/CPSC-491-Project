#!/usr/bin/env python
import sys
import os
sys.path.insert(0, '.')
os.environ['DJANGO_SETTINGS_MODULE'] = 'movie_recommendation.settings'

import django
django.setup()

from recommender.views import MovieRecommender

print('Starting model load...')
try:
    rec = MovieRecommender('models_fast')
    print('✓ SUCCESS! Model loaded.')
    print(f'✓ Loaded {rec.config["n_movies"]:,} movies')
    print(f'✓ Metadata shape: {rec.metadata.shape}')
    print(f'✓ Similarity matrix shape: {rec.similarity_matrix.shape}')
    print(f'✓ Title index size: {len(rec.title_to_idx)}')
except Exception as e:
    print(f'✗ ERROR: {e}')
    import traceback
    traceback.print_exc()
