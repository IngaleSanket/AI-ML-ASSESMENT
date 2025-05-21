# Install these packages before running:
# pip install sentence-transformers scikit-learn numpy

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import json

# Sample RTX 50 series laptop survey feedback
survey_responses = [
    "The RTX 50 laptop is insanely fast for gaming!",
    "Battery life could be better on full performance mode.",
    "I love how smooth the ray tracing looks now.",
    "It heats up pretty quickly under load.",
    "Perfect for 3D rendering and design work.",
    "Too bulky to carry around daily.",
    "Gaming performance is top-tier, even better than desktops.",
    "I wish it had more USB-C ports.",
    "Fans are loud when I'm gaming.",
    "The screen quality is stunning, crisp and vibrant.",
    "Thermals are a bit concerning after long sessions.",
    "Super fast boot and load times thanks to the SSD and GPU."
]

# Load offline model
print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Works fully offline once cached

# Encode feedback
print("📐 Encoding survey responses...")
embeddings = model.encode(survey_responses)

# Cluster into 3 insight themes
num_clusters = 3
print("🧠 Clustering into themes...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Group quotes by cluster
clustered = defaultdict(list)
for idx, label in enumerate(labels):
    clustered[label].append(survey_responses[idx])

# Simple keyword-based theme extractor
def extract_theme_name(quotes):
    keywords = {
        "battery": "Battery & Power",
        "performance": "Gaming Performance",
        "ray tracing": "Graphics & Ray Tracing",
        "thermals": "Heating & Cooling",
        "usb": "Connectivity",
        "fan": "Noise & Fans",
        "screen": "Display Quality",
        "boot": "Speed & Startup",
        "bulky": "Portability",
        "ssd": "Speed & Storage"
    }
    for quote in quotes:
        lower = quote.lower()
        for word, theme in keywords.items():
            if word in lower:
                return theme
    return "General Feedback"

# Build insight cards
insight_cards = []
for cluster_id, quotes in clustered.items():
    card = {
        "theme": extract_theme_name(quotes),
        "quotes": quotes,
        "sentiment": "neutral"  # Placeholder; you can add Vader sentiment later
    }
    insight_cards.append(card)

# Output result
print("\n✅ Final Insight Cards:")
for card in insight_cards:
    print("\n📌 Theme:", card["theme"])
    for quote in card["quotes"]:
        print(f"  - \"{quote}\"")

