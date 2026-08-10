"""
Mint Leaf AI — Agronomic Advisory Engine Module
Provides organic remedies, chemical treatments, and cultural controls per disease class.
"""

ADVISORY_DATABASE = {
    "Healthy": {
        "disease_name": "Healthy Mint Control (Mentha spp.)",
        "severity_level": "OPTIMAL_HEALTH",
        "summary": "No pathological infection or physiological deterioration detected.",
        "organic_treatment": "Apply balanced organic compost tea every 3-4 weeks to maintain soil microbiology.",
        "chemical_treatment": "No fungicide or chemical intervention required.",
        "cultural_controls": [
            "Maintain regular morning watering at the soil base.",
            "Prune dense canopy clusters to ensure adequate sunlight penetration.",
            "Inspect underside of leaves weekly for early pest signs."
        ]
    },
    "Mint_Rust": {
        "disease_name": "Mint Rust (Puccinia menthae)",
        "severity_level": "HIGH_SEVERITY_FUNGAL",
        "summary": "Fungal infection producing bright orange-yellow pustules on leaf undersides.",
        "organic_treatment": "Spray organic Neem Oil (0.5% concentration) or Potassium Bicarbonate solution.",
        "chemical_treatment": "Apply Copper Hydroxide or Propiconazole fungicide at recommended dosage.",
        "cultural_controls": [
            "Immediately clip and burn infected plant shoots; do NOT compost rust-infected tissue.",
            "Transition to drip irrigation to prevent water droplets on leaves.",
            "Sterilize pruning shears between plant cuts using 70% isopropyl alcohol."
        ]
    },
    "Powdery_Mildew": {
        "disease_name": "Powdery Mildew (Erysiphe cichoracearum)",
        "severity_level": "MODERATE_FUNGAL",
        "summary": "White-grey powdery fungal hyphae coating upper leaf surfaces.",
        "organic_treatment": "Apply Bio-fungicide (*Bacillus subtilis*) or diluted milk whey spray (1:9 ratio).",
        "chemical_treatment": "Spray Wettable Sulfur or Myclobutanil early upon symptom onset.",
        "cultural_controls": [
            "Increase plant spacing to improve air circulation across the crop canopy.",
            "Avoid high-nitrogen fertilizers that produce overly dense, susceptible foliage.",
            "Water early in the morning so sun dries leaf surfaces rapidly."
        ]
    },
    "Leaf_Spot": {
        "disease_name": "Septoria Leaf Spot (Septoria menthae)",
        "severity_level": "MODERATE_FUNGAL",
        "summary": "Small circular brown-black lesions with light centers leading to defoliation.",
        "organic_treatment": "Spray Copper Octanoate (Soap-based organic copper fungicide).",
        "chemical_treatment": "Apply Chlorothalonil or Mancozeb protective fungicide.",
        "cultural_controls": [
            "Remove lower infected leaves touching the soil.",
            "Apply straw mulch to prevent soil-borne fungal spores from splashing onto leaves.",
            "Rotate crop location every 2 years."
        ]
    },
    "Blight_Rhizoctonia": {
        "disease_name": "Leaf Blight & Rhizoctonia Rot",
        "severity_level": "CRITICAL_PATHOLOGY",
        "summary": "Large irregular water-soaked brown lesions causing rapid foliar necrosis.",
        "organic_treatment": "Apply Bio-fungicide (*Trichoderma harzianum*) to soil and foliar canopy.",
        "chemical_treatment": "Apply Azoxystrobin or Flutolanil targeted systemic fungicide.",
        "cultural_controls": [
            "Improve soil drainage immediately; avoid waterlogging root systems.",
            "Reduce irrigation frequency during cool, humid weather.",
            "Remove dead leaf litter around the base of mint plants."
        ]
    },
    "Post_Harvest_Deteriorated": {
        "disease_name": "Post-Harvest Deterioration / Spoilage",
        "severity_level": "PHYSIOLOGICAL_POST_HARVEST",
        "summary": "Foliar wilting, browning, and tissue breakdown post-harvest.",
        "organic_treatment": "Treat harvested leaves with food-grade citric acid wash (0.1%).",
        "chemical_treatment": "No post-harvest chemical synthetic sprays allowed on fresh herbs.",
        "cultural_controls": [
            "Maintain strict cold chain storage ($0 - 2^\\circ\\text{C}$ with $95\\%$ relative humidity).",
            "Use breathable perforated modified-atmosphere packaging (MAP).",
            "Pre-chill mint leaves within 2 hours of field harvest."
        ]
    }
}

def get_agronomic_advisory(disease_class):
    """
    Returns agronomic treatment recommendation card for the predicted disease class.
    """
    return ADVISORY_DATABASE.get(disease_class, ADVISORY_DATABASE["Healthy"])
