"""
=============================================================================
data_enrichment.py  —  Biagiotti  |  ML Data Enrichment Layer  v1.0
=============================================================================
Transforms a minimal CSV row (product_id, product_name, price, units_sold)
into a fully enriched product record so ALL downstream ML models receive
rich, realistic, product-unique inputs.

Ten sub-engines (each deterministic, seeded by product_id hash):
  1.  Product Understanding  → category, skin_target, function
  2.  Ingredient Generator   → realistic INCI ingredient string
  3.  Review Generator       → 5–15 synthetic reviews (40/30/30 split)
  4.  Skin Text Generator    → descriptive text for skin model
  5.  Demand Signal Enricher → trend, seasonality, shock factors
  6.  Uniqueness Enforcer    → product_hash, multipliers
  7.  Safety Auto-Classifier → ensures harmful detection varies
  8.  Sentiment Aggregator   → aggregated score from reviews
  9.  Final Assembler        → one clean enriched dict
  10. Output Validator       → guarantees no empty / duplicate fields
=============================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("data_enrichment")

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 1 — PRODUCT UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

# Maps product name keywords → category
_CATEGORY_MAP: Dict[str, List[str]] = {
    "serum":      ["serum", "ampoule", "concentrate", "shot", "booster", "essence"],
    "cleanser":   ["cleanser", "wash", "foaming", "micellar", "scrub", "exfoliant",
                   "exfoliator", "cleansing", "makeup remover", "peel"],
    "cream":      ["cream", "moisturizer", "moisturiser", "lotion", "balm",
                   "butter", "emulsion", "gel cream", "overnight", "day cream",
                   "night cream", "sleeping mask"],
    "sunscreen":  ["spf", "sunscreen", "sun protection", "solar", "uv", "sunblock"],
    "toner":      ["toner", "tonic", "mist", "facial water", "clarifying lotion"],
    "mask":       ["mask", "masque", "mud mask", "clay mask", "sheet mask",
                   "peel-off", "sleeping pack"],
    "oil":        ["face oil", "facial oil", "rosehip", "argan", "jojoba",
                   "beauty oil", "dry oil"],
    "eye":        ["eye cream", "eye gel", "eye serum", "eye balm",
                   "under eye", "dark circle"],
    "lip":        ["lip balm", "lip butter", "lip serum", "lip mask", "lip oil"],
}

# Maps keywords → skin target
_SKIN_TARGET_MAP: Dict[str, List[str]] = {
    "oily":      ["oily", "shine", "sebum control", "anti-shine", "matte",
                  "pore", "oil-free", "non-comedogenic"],
    "dry":       ["dry", "hydrat", "moisture", "nourish", "dehydrat",
                  "rich", "plump", "soothing"],
    "sensitive":  ["sensitive", "gentle", "calming", "soothing", "hypoallerg",
                   "fragrance-free", "mild"],
    "acne":      ["acne", "blemish", "pimple", "spot", "breakout",
                  "salicylic", "benzoyl"],
    "anti-aging": ["anti-aging", "anti-ageing", "wrinkle", "firming", "lifting",
                   "retinol", "peptide", "collagen"],
    "brightening":["bright", "glow", "vitamin c", "radiance", "dark spot",
                   "even tone", "luminous"],
    "normal":    [],  # fallback
}

# Maps keywords → product function
_FUNCTION_MAP: Dict[str, List[str]] = {
    "hydration":      ["hydrat", "moisture", "plump", "water", "hyaluronic"],
    "acne-control":   ["acne", "blemish", "pimple", "salicylic", "benzoyl"],
    "brightening":    ["bright", "vitamin c", "glow", "radiance", "pigment"],
    "anti-aging":     ["anti-aging", "anti-ageing", "retinol", "peptide",
                       "wrinkle", "firming"],
    "protection":     ["spf", "sunscreen", "uv", "sun"],
    "cleansing":      ["clean", "wash", "remove", "purif", "detox"],
    "soothing":       ["calm", "sooth", "gentle", "sensitive", "aloe"],
    "pore-control":   ["pore", "sebum", "matte", "oily", "non-comedogenic"],
    "nourishment":    ["nourish", "oil", "butter", "ceramide", "rich"],
    "exfoliation":    ["exfoliat", "scrub", "aha", "bha", "peel", "acid"],
}


def infer_product_understanding(name: str) -> Dict[str, str]:
    """Engine 1: Infer category, skin_target, function from product name."""
    n = name.lower()

    # ── Category ─────────────────────────────────────────────────────────────
    category = "general"
    for cat, keywords in _CATEGORY_MAP.items():
        if any(kw in n for kw in keywords):
            category = cat
            break

    # ── Skin target ───────────────────────────────────────────────────────────
    skin_target = "normal"
    for target, keywords in _SKIN_TARGET_MAP.items():
        if keywords and any(kw in n for kw in keywords):
            skin_target = target
            break

    # ── Function ─────────────────────────────────────────────────────────────
    function = "general-care"
    for func, keywords in _FUNCTION_MAP.items():
        if any(kw in n for kw in keywords):
            function = func
            break

    return {"category": category, "skin_target": skin_target, "function": function}


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 2 — INGREDIENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MEGA INGREDIENT DATASET — 50+ INCI ingredients per category
# Sourced from real cosmetic formulation databases (EU CosIng, EWG, INCIDecoder)
# ─────────────────────────────────────────────────────────────────────────────
_INGREDIENT_BASES: Dict[str, List[str]] = {
    "serum": [
        # Water base
        "Aqua (Water)", "Rosa Damascena Flower Water", "Centella Asiatica Water",
        # Humectants
        "Glycerin", "Sodium Hyaluronate", "Hyaluronic Acid", "Sodium PCA",
        "Trehalose", "Betaine", "Urea (5%)", "Propanediol", "Panthenol",
        # Active ingredients
        "Niacinamide (10%)", "Niacinamide (5%)", "Ascorbic Acid (Vitamin C)",
        "Sodium Ascorbyl Phosphate", "Magnesium Ascorbyl Phosphate",
        "Alpha Arbutin", "Tranexamic Acid", "Azelaic Acid",
        "Retinol (0.1%)", "Retinol (0.5%)", "Bakuchiol", "Peptide Complex",
        "Argireline (Acetyl Hexapeptide-3)", "Palmitoyl Tripeptide-1",
        "Matrixyl 3000 (Palmitoyl Oligopeptide)", "Copper Peptide (GHK-Cu)",
        "Epidermal Growth Factor (EGF)", "Resveratrol",
        # Soothing & repair
        "Centella Asiatica Extract", "Madecassoside", "Asiaticoside",
        "Allantoin", "Bisabolol", "Beta-Glucan", "Aloe Barbadensis Leaf Juice",
        "Oat Kernel Extract", "Licorice Root Extract", "Green Tea Extract (EGCG)",
        # Ceramides & barrier
        "Ceramide NP", "Ceramide AP", "Phytosphingosine",
        # Antioxidants
        "Ferulic Acid", "Vitamin E (Tocopherol)", "Astaxanthin",
        "Coenzyme Q10 (Ubiquinone)", "Superoxide Dismutase",
        # Amino acids
        "Arginine", "Proline", "Lysine", "Serine",
        # Functional
        "Zinc PCA", "Adenosine", "Acetyl Glucosamine",
        # Solvents/carriers
        "1,2-Hexanediol", "Caprylyl Glycol",
    ],

    "cleanser": [
        # Water base
        "Aqua (Water)",
        # Surfactants (mild/gentle)
        "Sodium Cocoyl Isethionate", "Cocamidopropyl Betaine",
        "Sodium Lauroyl Sarcosinate", "Coco-Glucoside",
        "Disodium Cocoamphodiacetate", "Decyl Glucoside",
        "Lauryl Glucoside", "Sodium Methyl Cocoyl Taurate",
        "Sodium Cocoyl Glutamate", "Potassium Cocoate",
        # Harsh surfactants (can appear in budget products)
        "Sodium Lauryl Sulfate", "Sodium Laureth Sulfate",
        # Humectants
        "Glycerin", "Sorbitol", "Propanediol", "Sodium PCA",
        # Active ingredients
        "Salicylic Acid (0.5%)", "Salicylic Acid (2%)",
        "Glycolic Acid", "Lactic Acid (5%)", "Gluconolactone",
        "Mandelic Acid", "Niacinamide", "Zinc PCA",
        # Soothing
        "Allantoin", "Aloe Vera Gel", "Chamomile Extract",
        "Centella Asiatica Extract", "Panthenol", "Bisabolol",
        "Calendula Extract", "Cucumber Extract",
        # Botanicals
        "Green Tea Extract", "Witch Hazel Extract",
        "Willow Bark Extract", "Tea Tree Oil",
        # Emollients
        "Jojoba Oil", "Squalane", "Caprylic/Capric Triglyceride",
        # Thickeners
        "Carbomer", "Xanthan Gum", "Acrylates/C10-30 Alkyl Acrylate Crosspolymer",
        "Cellulose Gum", "Hydroxyethylcellulose",
        # Foam boosters
        "Cocamide DEA", "Lauramide DEA",
    ],

    "cream": [
        # Water phase
        "Aqua (Water)",
        # Emollients & oils
        "Glycerin", "Squalane", "Caprylic/Capric Triglyceride",
        "Jojoba Oil", "Shea Butter", "Mango Seed Butter",
        "Cocoa Butter", "Sweet Almond Oil", "Rosehip Seed Oil",
        "Sunflower Seed Oil", "Evening Primrose Oil",
        "Mineral Oil", "Dimethicone", "Cyclomethicone",
        # Emulsifiers
        "Cetyl Alcohol", "Stearyl Alcohol", "Cetearyl Alcohol",
        "Glyceryl Stearate", "PEG-100 Stearate", "Polysorbate 60",
        "Sorbitan Stearate", "Lecithin",
        # Barrier repair
        "Ceramide NP", "Ceramide AP", "Ceramide EOP", "Ceramide EOS",
        "Cholesterol", "Phytosphingosine", "Sphingolipids",
        # Occlusives
        "Petrolatum", "Beeswax", "Microcrystalline Wax",
        # Humectants
        "Sodium Hyaluronate", "Urea (10%)", "Sorbitol", "Betaine",
        # Actives
        "Niacinamide (5%)", "Retinol (0.025%)", "Retinol (0.1%)",
        "Retinaldehyde", "Bakuchiol", "Peptide Complex",
        "Palmitoyl Tripeptide-1", "Adenosine",
        "Tranexamic Acid", "Azelaic Acid (5%)",
        # Soothing
        "Panthenol", "Allantoin", "Bisabolol",
        "Centella Asiatica Extract", "Oat Kernel Extract",
        # Antioxidants
        "Tocopheryl Acetate (Vitamin E)", "Ferulic Acid",
        "Vitamin C (Ascorbic Acid)", "Resveratrol",
        # Thickeners/stabilisers
        "Carbomer", "Xanthan Gum", "Hydroxyethylcellulose",
        "PEG-20M", "Cellulose",
    ],

    "sunscreen": [
        # Water base
        "Aqua (Water)", "Glycerin", "Niacinamide",
        # Chemical UV filters
        "Homosalate (10%)", "Octisalate (5%)", "Octocrylene (7%)",
        "Avobenzone (3%)", "Oxybenzone (6%)", "Octinoxate (7.5%)",
        "Benzophenone-4", "Ethylhexyl Methoxycinnamate",
        # Physical UV filters (mineral)
        "Zinc Oxide (20%)", "Zinc Oxide (10%)", "Titanium Dioxide (5%)",
        "Titanium Dioxide (2%)",
        # Boosters
        "Phenylbenzimidazole Sulfonic Acid", "Bis-Ethylhexyloxyphenol Methoxyphenyl Triazine",
        "Diethylamino Hydroxybenzoyl Hexyl Benzoate",
        # Emollients
        "C12-15 Alkyl Benzoate", "Dimethicone", "Squalane",
        "Caprylic/Capric Triglyceride", "Isononyl Isononanoate",
        "Cetyl Dimethicone", "Silica",
        # Emulsifiers
        "Cyclopentasiloxane", "PEG-9 Polydimethylsiloxyethyl Dimethicone",
        "Glyceryl Stearate", "PEG-100 Stearate",
        # Skin benefits
        "Niacinamide (5%)", "Hyaluronic Acid", "Centella Asiatica Extract",
        "Panthenol", "Vitamin E (Tocopherol)", "Green Tea Extract",
        "Aloe Barbadensis Leaf Juice",
        # Thickeners / film formers
        "Acrylates/C10-30 Alkyl Acrylate Crosspolymer", "Carbomer",
        "Polymethyl Methacrylate",
        # Antioxidants
        "BHT", "BHA",
    ],

    "toner": [
        # Water base
        "Aqua (Water)", "Rosa Damascena Flower Water",
        "Centella Asiatica Water", "Witch Hazel Water",
        "Cucumber Distillate",
        # Humectants
        "Glycerin", "Propanediol", "Butylene Glycol",
        "Sodium Hyaluronate", "Hyaluronic Acid", "Sodium PCA",
        "Trehalose", "Panthenol",
        # Exfoliating acids
        "Glycolic Acid (5%)", "Glycolic Acid (10%)", "Lactic Acid (5%)",
        "Mandelic Acid (5%)", "Gluconolactone", "Polyhydroxy Acid",
        "Tartaric Acid", "Citric Acid",
        # Soothing actives
        "Niacinamide", "Centella Asiatica Extract", "Allantoin",
        "Bisabolol", "Aloe Barbadensis Leaf Juice",
        "Green Tea Extract", "Licorice Root Extract",
        # Astringents
        "Witch Hazel Extract", "Zinc PCA", "Salicylic Acid (0.5%)",
        # Botanical extracts
        "Chamomile Extract", "Calendula Extract", "Cucumber Extract",
        "Camellia Sinensis Leaf Extract", "Fermented Rice Water",
        # Repair actives
        "Beta-Glucan", "Oat Kernel Extract", "Ceramide NP",
        # Brightening
        "Alpha Arbutin", "Tranexamic Acid", "Vitamin C Derivative",
        # Functional
        "4-Butylresorcinol", "Azelaic Acid", "Kojic Acid",
    ],

    "mask": [
        "Aqua (Water)", "Glycerin",
        # Clay / absorbing agents
        "Kaolin", "Bentonite", "Montmorillonite Clay",
        "Zeolite", "Activated Charcoal", "Charcoal Powder",
        "Bamboo Powder", "Diatomaceous Earth",
        # Moisturising / soothing
        "Aloe Vera Gel (99%)", "Sodium Hyaluronate", "Hyaluronic Acid",
        "Ceramide NP", "Colloidal Oatmeal", "Centella Asiatica Extract",
        "Centella Asiatica Leaf Extract", "Panthenol",
        "Allantoin", "Chamomile Extract", "Rose Water",
        # Actives
        "Salicylic Acid (1%)", "Tea Tree Oil", "Zinc Oxide",
        "Niacinamide", "Retinol", "Glycolic Acid (5%)", "Lactic Acid",
        "Sulfur (3%)", "Mandelic Acid",
        # Brightening
        "Vitamin C (Sodium Ascorbyl Phosphate)", "Alpha Arbutin",
        "Turmeric Extract", "Licorice Root Extract",
        # Emollients
        "Squalane", "Jojoba Oil", "Sweet Almond Oil",
        # Sheet mask carriers
        "Propanediol", "Butylene Glycol", "Betaine",
        # Thickeners
        "Carbomer", "Xanthan Gum", "Hydroxyethylcellulose",
    ],

    "oil": [
        "Squalane", "Jojoba Oil (Simmondsia Chinensis Seed Oil)",
        "Rosehip Seed Oil (Rosa Canina Fruit Oil)",
        "Argan Oil (Argania Spinosa Kernel Oil)",
        "Marula Oil (Sclerocarya Birrea Seed Oil)",
        "Sea Buckthorn Oil (Hippophae Rhamnoides Oil)",
        "Chia Seed Oil (Salvia Hispanica Seed Oil)",
        "Hemp Seed Oil (Cannabis Sativa Seed Oil)",
        "Black Seed Oil (Nigella Sativa Seed Oil)",
        "Broccoli Seed Oil (Brassica Oleracea Italica Seed Oil)",
        "Buriti Oil (Mauritia Flexuosa Fruit Oil)",
        "Cacay Oil (Caryodendron Orinocense Seed Oil)",
        "Carrot Seed Oil (Daucus Carota Sativa Seed Oil)",
        "Camellia Seed Oil (Camellia Oleifera Seed Oil)",
        "Evening Primrose Oil (Oenothera Biennis Oil)",
        "Pomegranate Seed Oil (Punica Granatum Seed Oil)",
        "Prickly Pear Seed Oil (Opuntia Ficus-Indica Seed Oil)",
        "Sunflower Seed Oil (Helianthus Annuus Seed Oil)",
        "Bakuchiol (Psoralea Corylifolia Seed Extract)",
        "Vitamin E (Tocopherol)", "Vitamin E Acetate (Tocopheryl Acetate)",
        "Vitamin A Palmitate", "Retinol", "Beta-Carotene",
        "Coenzyme Q10 (Ubiquinone)",
        "Astaxanthin", "Lycopene",
        "Lavender Essential Oil", "Frankincense Essential Oil",
        "Geranium Essential Oil",
    ],

    "eye": [
        "Aqua (Water)", "Glycerin",
        # Puffiness/dark circles
        "Caffeine (3%)", "Vitamin K (Phylloquinone)",
        "Dipeptide-2", "Haloxyl (Palmitoyl Tetrapeptide-7)",
        # Firming peptides
        "Peptide Complex (Argireline)", "Palmitoyl Tripeptide-1",
        "Acetyl Hexapeptide-3 (Argireline)", "Leuphasyl",
        # Hydration
        "Sodium Hyaluronate", "Hyaluronic Acid", "Sodium PCA",
        "Panthenol", "Glycerin", "Aloe Vera",
        # Barrier
        "Ceramide NP", "Shea Butter", "Squalane",
        # Brightening
        "Niacinamide (5%)", "Alpha Arbutin", "Vitamin C Derivative",
        # Soothing
        "Centella Asiatica Extract", "Allantoin", "Bisabolol",
        "Cucumber Extract", "Chamomile Extract",
        # Retinoids (gentle forms)
        "Retinol (0.025%)", "Retinaldehyde", "Bakuchiol",
        # Antioxidants
        "Vitamin E (Tocopherol)", "Coenzyme Q10", "Resveratrol",
        # Emollients
        "Dimethicone", "Caprylic/Capric Triglyceride",
        # Thickeners
        "Carbomer", "Xanthan Gum",
    ],

    "lip": [
        "Petrolatum", "Beeswax (Cera Alba)", "Candelilla Wax",
        "Carnauba Wax", "Ozokerite", "Microcrystalline Wax",
        "Castor Oil (Ricinus Communis Seed Oil)",
        "Shea Butter (Butyrospermum Parkii)",
        "Coconut Oil (Cocos Nucifera Oil)",
        "Jojoba Oil (Simmondsia Chinensis Seed Oil)",
        "Sweet Almond Oil", "Avocado Oil (Persea Gratissima Oil)",
        "Lanolin", "Hydrogenated Lanolin",
        "Vitamin E (Tocopheryl Acetate)",
        "Vitamin E (Tocopherol)", "Vitamin C Derivative",
        "Niacinamide (2%)", "Ceramide NP",
        "Hyaluronic Acid", "Sodium Hyaluronate",
        "Aloe Vera Extract (Aloe Barbadensis Leaf Extract)",
        "Peptide Complex", "Argireline",
        "Peppermint Oil", "Menthol",
        "Vanilla Planifolia Fruit Extract",
        "Raspberry Seed Oil", "Pomegranate Extract",
        "D-Panthenol", "Allantoin",
        "Caprylic/Capric Triglyceride", "Squalane",
        "Isopropyl Myristate", "Isopropyl Palmitate",
    ],

    "general": [
        "Aqua (Water)", "Glycerin", "Niacinamide", "Hyaluronic Acid",
        "Panthenol", "Allantoin", "Ceramide NP", "Squalane",
        "Dimethicone", "Xanthan Gum", "Vitamin E (Tocopherol)",
        "Centella Asiatica Extract", "Green Tea Extract",
        "Aloe Barbadensis Leaf Juice", "Sodium Hyaluronate",
        "Caprylic/Capric Triglyceride", "Carbomer",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD-SPECIFIC INGREDIENT BOOSTING
# When a product name contains these keywords, add these signature ingredients
# on top of the category base — this is the "mega dataset aware" prediction
# ─────────────────────────────────────────────────────────────────────────────
_KEYWORD_BOOST_INGREDIENTS: Dict[str, List[str]] = {
    # Active ingredients by name
    "retinol":       ["Retinol (0.1%)", "Retinol (0.5%)", "Retinaldehyde", "Bakuchiol"],
    "vitamin c":     ["Ascorbic Acid (Vitamin C)", "Sodium Ascorbyl Phosphate",
                      "Magnesium Ascorbyl Phosphate", "Ascorbyl Glucoside", "Ferulic Acid"],
    "niacinamide":   ["Niacinamide (10%)", "Zinc PCA", "Sodium Hyaluronate"],
    "hyaluronic":    ["Hyaluronic Acid (Multi-weight)", "Sodium Hyaluronate (Low MW)",
                      "Sodium Hyaluronate (High MW)", "Hydrolyzed Hyaluronic Acid"],
    "salicylic":     ["Salicylic Acid (2%)", "Zinc PCA", "Tea Tree Oil", "Willow Bark Extract"],
    "glycolic":      ["Glycolic Acid (10%)", "Glycolic Acid (5%)", "Tartaric Acid", "Citric Acid"],
    "lactic":        ["Lactic Acid (10%)", "Lactic Acid (5%)", "Willow Bark Extract"],
    "aha":           ["Glycolic Acid", "Lactic Acid", "Mandelic Acid", "Tartaric Acid"],
    "bha":           ["Salicylic Acid (2%)", "Willow Bark Extract", "Zinc PCA"],
    "peptide":       ["Palmitoyl Tripeptide-1", "Acetyl Hexapeptide-3",
                      "Palmitoyl Oligopeptide", "Matrixyl 3000", "Copper Peptide (GHK-Cu)"],
    "collagen":      ["Hydrolyzed Collagen", "Palmitoyl Tripeptide-1",
                      "Glycine", "Proline", "Hydroxyproline"],
    "ceramide":      ["Ceramide NP", "Ceramide AP", "Ceramide EOP",
                      "Ceramide EOS", "Ceramide NS", "Phytosphingosine", "Cholesterol"],
    "azelaic":       ["Azelaic Acid (10%)", "Azelaic Acid (20%)", "Niacinamide"],
    "brightening":   ["Alpha Arbutin (2%)", "Tranexamic Acid", "Kojic Acid",
                      "Licorice Root Extract", "Vitamin C Derivative"],
    "glow":          ["Vitamin C (Ascorbic Acid)", "Alpha Arbutin",
                      "Niacinamide (10%)", "Glycolic Acid"],
    "acne":          ["Salicylic Acid (2%)", "Niacinamide (10%)", "Zinc PCA",
                      "Tea Tree Oil", "Benzoyl Peroxide", "Willow Bark Extract"],
    "anti-aging":    ["Retinol (0.1%)", "Peptide Complex", "Adenosine",
                      "Coenzyme Q10", "Vitamin C", "Ferulic Acid", "Bakuchiol"],
    "anti-ageing":   ["Retinol (0.1%)", "Peptide Complex", "Adenosine",
                      "Coenzyme Q10", "Vitamin C", "Ferulic Acid"],
    "spf":           ["Zinc Oxide (20%)", "Titanium Dioxide (5%)",
                      "Avobenzone (3%)", "Octinoxate (7.5%)", "Niacinamide"],
    "sunscreen":     ["Zinc Oxide (20%)", "Titanium Dioxide (5%)",
                      "Octisalate (5%)", "Octocrylene (7%)"],
    "snail":         ["Snail Secretion Filtrate (96%)", "Glycoprotein",
                      "Allantoin", "Collagen"],
    "rosehip":       ["Rosehip Seed Oil", "Vitamin A Palmitate",
                      "Linoleic Acid", "Vitamin E (Tocopherol)"],
    "bakuchiol":     ["Bakuchiol (2%)", "Vitamin E (Tocopherol)", "Squalane"],
    "centella":      ["Centella Asiatica Extract (70%)", "Madecassoside",
                      "Asiaticoside", "Asiatic Acid", "Madecassic Acid"],
    "arbutin":       ["Alpha Arbutin (2%)", "Beta Arbutin", "Niacinamide",
                      "Tranexamic Acid"],
    "caffeine":      ["Caffeine (3%)", "Green Tea Extract", "EGCG"],
    "charcoal":      ["Activated Charcoal", "Kaolin", "Bentonite",
                      "Salicylic Acid (1%)"],
    "clay":          ["Kaolin", "Bentonite", "Montmorillonite Clay",
                      "Zeolite", "Sulfur"],
    "exfoliat":      ["Glycolic Acid (10%)", "Lactic Acid (5%)",
                      "Mandelic Acid", "Gluconolactone", "Polyhydroxy Acid"],
    "micellar":      ["Poloxamer 184", "Glycerin", "Centella Asiatica Extract"],
    "eye":           ["Caffeine (3%)", "Vitamin K", "Dipeptide-2",
                      "Sodium Hyaluronate", "Haloxyl"],
    "lip":           ["Shea Butter", "Castor Oil", "Ceramide NP",
                      "Vitamin E", "Hyaluronic Acid"],
    "sensitive":     ["Centella Asiatica Extract", "Allantoin", "Bisabolol",
                      "Oat Kernel Extract", "Panthenol"],
    "hydrat":        ["Hyaluronic Acid (Multi-weight)", "Trehalose",
                      "Sodium PCA", "Panthenol", "Betaine"],
    "moisture":      ["Ceramide NP", "Sodium Hyaluronate",
                      "Squalane", "Glycerin", "Urea (5%)"],
    "firming":       ["Palmitoyl Tripeptide-1", "Adenosine",
                      "Acetyl Hexapeptide-3", "DMAE", "Argireline"],
    "pore":          ["Niacinamide (10%)", "Zinc PCA", "Salicylic Acid",
                      "Kaolin", "Silica"],
    "dark spot":     ["Alpha Arbutin (2%)", "Tranexamic Acid",
                      "Kojic Acid", "Vitamin C", "Niacinamide"],
    "dark circle":   ["Caffeine (3%)", "Vitamin K", "Dipeptide-2",
                      "Haloxyl", "Alpha Arbutin"],
    "oily":          ["Niacinamide (10%)", "Zinc PCA", "Salicylic Acid",
                      "Kaolin", "Green Tea Extract"],
    "dry":           ["Hyaluronic Acid", "Ceramide NP", "Shea Butter",
                      "Squalane", "Urea (5%)"],
    "redness":       ["Centella Asiatica Extract", "Bisabolol", "Azelaic Acid",
                      "Allantoin", "Chamomile Extract"],
    "repair":        ["Ceramide NP", "Ceramide AP", "Panthenol",
                      "Centella Asiatica Extract", "Madecassoside"],
    "barrier":       ["Ceramide NP", "Ceramide AP", "Cholesterol",
                      "Fatty Acids", "Phytosphingosine"],
}

# Function-specific boosting ingredients
_FUNCTION_INGREDIENTS: Dict[str, List[str]] = {
    "acne-control":  ["Salicylic Acid (2%)", "Benzoyl Peroxide", "Tea Tree Oil",
                      "Zinc PCA", "Niacinamide (10%)"],
    "brightening":   ["Vitamin C (Ascorbic Acid)", "Alpha Arbutin", "Kojic Acid",
                      "Tranexamic Acid", "Licorice Root Extract"],
    "anti-aging":    ["Retinol", "Peptide Complex", "Adenosine", "Bakuchiol",
                      "CoQ10", "Vitamin A Palmitate"],
    "hydration":     ["Hyaluronic Acid (Multi-weight)", "Sodium PCA",
                      "Trehalose", "Glycerin", "Aloe Barbadensis Leaf Extract"],
    "soothing":      ["Aloe Vera (99%)", "Centella Asiatica Extract",
                      "Chamomile Extract", "Madecassoside", "Bisabolol"],
    "pore-control":  ["Niacinamide (10%)", "Zinc PCA", "Kaolin", "Salicylic Acid"],
}

# Risk ingredients: (ingredient_str, risk_label)
_RISK_POOL: List[Tuple[str, str]] = [
    ("Methylparaben",          "paraben"),
    ("Propylparaben",          "paraben"),
    ("Ethylparaben",           "paraben"),
    ("Fragrance (Parfum)",     "fragrance"),
    ("Denatured Alcohol",      "alcohol"),
    ("SD Alcohol 40",          "alcohol"),
    ("Oxybenzone",             "oxybenzone"),
    ("Formaldehyde",           "formaldehyde"),
    ("Sodium Lauryl Sulfate",  "sls"),
    ("Benzoyl Peroxide",       "benzoyl_peroxide"),
    ("Hydroquinone",           "hydroquinone"),
]

# Tail-end safe excipients
_EXCIPIENTS: List[str] = [
    "Phenoxyethanol", "Ethylhexylglycerin", "Sodium Benzoate",
    "Potassium Sorbate", "Citric Acid", "Sodium Hydroxide",
    "Carbomer", "Xanthan Gum", "Disodium EDTA",
    "Triethanolamine", "Tocopherol (Vitamin E)",
]


def generate_ingredients(
    category: str,
    skin_target: str,
    function: str,
    rng: random.Random,
    product_name: str = "",
) -> Tuple[str, List[str]]:
    """
    Engine 2: Generate a realistic INCI ingredient list using the mega-dataset.

    Strategy:
      1. Pull 10-18 base ingredients from the category mega-dataset.
      2. Boost with function-specific signature actives (3-5 ingredients).
      3. Boost with product-name keyword actives (0-4 ingredients).
      4. Inject risk ingredients probabilistically (18% per slot).
      5. Add 3-5 safe tail excipients.

    Returns (ingredient_string, list_of_risk_labels).
    """
    name_lower = product_name.lower()

    # 1. Category base (pick 10-18 from the mega-dataset)
    base = list(_INGREDIENT_BASES.get(category, _INGREDIENT_BASES["general"]))
    n_base = min(len(base), rng.randint(10, 18))
    chosen: List[str] = rng.sample(base, n_base)

    # 2. Function-specific signature actives
    func_ings = list(_FUNCTION_INGREDIENTS.get(function, []))
    if func_ings:
        chosen += rng.sample(func_ings, min(len(func_ings), rng.randint(2, 4)))

    # 3. Keyword-driven ingredient boosting from product name (mega-dataset aware)
    keyword_ings: List[str] = []
    for kw, boost_list in _KEYWORD_BOOST_INGREDIENTS.items():
        if kw in name_lower:
            # Take 1–3 signature ingredients for this keyword
            picks = rng.sample(boost_list, min(len(boost_list), rng.randint(1, 3)))
            keyword_ings.extend(picks)
    # Add up to 5 unique keyword boost ingredients at the front (high-priority)
    chosen = keyword_ings[:5] + chosen

    risks: List[str] = []

    # 4. Risk injection (probabilistic) — ~18% chance per slot
    for item, label in rng.sample(_RISK_POOL, min(len(_RISK_POOL), 5)):
        if rng.random() < 0.18:
            chosen.append(item)
            risks.append(label)

    # Acne skin target has higher chance of benzoyl peroxide
    if skin_target == "acne" and rng.random() < 0.35:
        chosen.append("Benzoyl Peroxide")
        risks.append("benzoyl_peroxide")

    # 5. Add 3-5 safe excipients at the tail
    excipients = rng.sample(_EXCIPIENTS, rng.randint(3, 5))
    chosen += excipients

    # Deduplicate while preserving insertion order
    seen: set = set()
    final: List[str] = []
    for ing in chosen:
        key = re.sub(r"\s*\(.*?\)", "", ing.lower()).strip()   # strip % / percentages for dedup
        if key not in seen:
            seen.add(key)
            final.append(ing)

    return ", ".join(final), risks


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 3 — REVIEW GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_REVIEW_TEMPLATES: Dict[str, List[str]] = {
    "positive": [
        "Absolutely love this {name}! My {skin} skin has never looked better.",
        "This {name} is a complete game changer. Noticed results in just one week!",
        "Best {cat} I have ever tried. Lightweight, non-greasy, and very effective.",
        "My dermatologist recommended this and it really works. No more {issue}.",
        "Incredible results! Skin feels so smooth and {benefit} after regular use.",
        "Been using this for 3 weeks — the {benefit} effect is genuinely visible.",
        "Packaging is premium and the formula is even better. Highly recommend!",
        "Finally found something that works for my {skin} skin. This is it.",
        "Five stars, no hesitation. Will definitely repurchase.",
        "The texture is divine — absorbs instantly and leaves no residue.",
    ],
    "neutral": [
        "It's decent but nothing extraordinary. Does the job for daily use.",
        "Average product. Works okay but did not see dramatic results.",
        "Price feels a little high for what it delivers. Formula is fine though.",
        "Not bad, not amazing. Texture is nice but results are underwhelming.",
        "I expected more from this {name}. It's mediocre at best.",
        "Some improvement in {benefit} but very subtle. Takes too long.",
        "Packaging is nice. Formula is standard. Results are average.",
        "Would use again if on sale but would not pay full price.",
        "Three out of five. Does what it claims, nothing more.",
        "My skin did not react badly but I also do not see visible change.",
    ],
    "negative": [
        "Caused breakouts after the second application. Very disappointed.",
        "Broke out all over my face. The {issue} ingredient probably caused it.",
        "Irritated my {skin} skin. Returned it immediately.",
        "Too greasy and heavy. Sits on top of the skin instead of absorbing.",
        "Smells chemical and synthetic. Definitely has fragrance despite claims.",
        "Gave me redness and itching within 24 hours of use. Avoid.",
        "Complete waste of money. No visible difference after one month.",
        "The formula separated after two weeks. Quality control issue.",
        "Clogs my pores and caused milia around my eye area.",
        "Tried for 6 weeks and got zero results. Very disappointed.",
    ],
}

_SKIN_WORDS = {
    "oily": "oily",
    "dry": "dry",
    "sensitive": "sensitive",
    "acne": "acne-prone",
    "anti-aging": "mature",
    "brightening": "dull",
    "normal": "combination",
    "general": "normal",
}

_ISSUE_WORDS = {
    "acne-control": "acne",
    "brightening": "dark spots",
    "anti-aging": "wrinkles",
    "hydration": "dehydration",
    "soothing": "redness",
    "pore-control": "enlarged pores",
    "cleansing": "clogged pores",
    "exfoliation": "dullness",
    "general-care": "dryness",
    "protection": "sun damage",
    "nourishment": "dryness",
}

_BENEFIT_WORDS = {
    "acne-control":  "clearer",
    "brightening":   "glowing",
    "anti-aging":    "firmer",
    "hydration":     "plumped",
    "soothing":      "calmed",
    "pore-control":  "refined",
    "cleansing":     "clean",
    "exfoliation":   "smooth",
    "general-care":  "healthier",
    "protection":    "protected",
    "nourishment":   "nourished",
}


def generate_reviews(
    product_name: str,
    category: str,
    skin_target: str,
    function: str,
    risk_labels: List[str],
    rng: random.Random,
) -> Tuple[List[str], float]:
    """
    Engine 3: Generate 5–15 synthetic reviews.
    Returns (list_of_review_strings, aggregate_sentiment_score).
    Sentiment score: -1.0 (very negative) → +1.0 (very positive).
    """
    short_name = product_name.split()[0] if product_name else "product"
    skin_w  = _SKIN_WORDS.get(skin_target, "normal")
    issue_w = _ISSUE_WORDS.get(function, "dryness")
    benefit_w = _BENEFIT_WORDS.get(function, "healthier")
    cat_w   = category if category != "general" else "skincare product"

    def _fill(template: str) -> str:
        return (template
                .replace("{name}", short_name)
                .replace("{skin}", skin_w)
                .replace("{issue}", issue_w)
                .replace("{benefit}", benefit_w)
                .replace("{cat}", cat_w))

    # Products with risk ingredients lean more negative
    if risk_labels:
        weights = {"positive": 25, "neutral": 30, "negative": 45}
    else:
        weights = {"positive": 45, "neutral": 30, "negative": 25}

    n_reviews = rng.randint(5, 14)
    reviews: List[str] = []
    sentiment_sum = 0.0

    pool = (
        [("positive", +1.0)] * weights["positive"] +
        [("neutral",   0.0)] * weights["neutral"] +
        [("negative", -1.0)] * weights["negative"]
    )

    for _ in range(n_reviews):
        tone, score = rng.choice(pool)
        template = rng.choice(_REVIEW_TEMPLATES[tone])
        reviews.append(_fill(template))
        # Add slight noise so identical templates score differently
        sentiment_sum += score + rng.uniform(-0.1, 0.1)

    agg_score = round(sentiment_sum / max(1, n_reviews), 3)
    return reviews, max(-1.0, min(1.0, agg_score))


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 4 — SKIN TEXT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_SKIN_TEXT_TEMPLATES: List[str] = [
    "This {adj} {cat} is ideal for {skin} skin. "
    "It {action} and helps achieve a visibly {benefit} complexion over time.",
    "Formulated for {skin} skin types, this {cat} works to {action}. "
    "Designed for daily use to deliver {benefit} results.",
    "A lightweight {cat} crafted for those with {skin} skin. "
    "The formula {action}, making it perfect for {benefit} results.",
    "Targeting {skin} skin concerns, this {cat} {action}. "
    "Regular use promotes a {benefit} and even skin tone.",
    "Clinically developed for {skin} skin, this {cat} {action}. "
    "Expect noticeably {benefit} skin within 4 weeks of consistent use.",
]

_ADJ_BY_CAT = {
    "serum": "concentrated",
    "cleanser": "gentle foaming",
    "cream": "rich nourishing",
    "sunscreen": "broad-spectrum",
    "toner": "balancing",
    "mask": "deep-action",
    "oil": "nourishing facial",
    "eye": "targeted",
    "lip": "hydrating",
    "general": "multi-action",
}

_ACTION_BY_FUNCTION = {
    "hydration":      "deeply hydrates and locks in moisture",
    "acne-control":   "reduces breakouts and controls excess sebum",
    "brightening":    "fades dark spots and enhances radiance",
    "anti-aging":     "minimises fine lines and firms the skin",
    "protection":     "shields against UV damage and pollution",
    "cleansing":      "removes impurities without stripping natural oils",
    "soothing":       "calms redness and strengthens the skin barrier",
    "pore-control":   "minimises pores and controls shine all day",
    "nourishment":    "deeply nourishes and restores skin softness",
    "exfoliation":    "gently exfoliates and reveals fresh skin",
    "general-care":   "supports overall skin health and hydration",
}


def generate_skin_text(
    product_name: str,
    category: str,
    skin_target: str,
    function: str,
    rng: random.Random,
) -> str:
    """Engine 4: Generate descriptive text that feeds the skin-type model."""
    template = rng.choice(_SKIN_TEXT_TEMPLATES)
    return template.replace(
        "{adj}",     _ADJ_BY_CAT.get(category, "multi-action")
    ).replace(
        "{cat}",     category if category != "general" else "skincare product"
    ).replace(
        "{skin}",    _SKIN_WORDS.get(skin_target, "normal")
    ).replace(
        "{action}",  _ACTION_BY_FUNCTION.get(function, "supports skin health")
    ).replace(
        "{benefit}", _BENEFIT_WORDS.get(function, "healthier")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 5 — DEMAND SIGNAL ENRICHER
# ─────────────────────────────────────────────────────────────────────────────

# Category → peak month index (0=Jan), seasonality type
_SEASONALITY_PROFILES: Dict[str, Dict] = {
    "sunscreen":  {"peak_month": 5,  "type": "strong_seasonal",  "amplitude": 0.40},
    "cream":      {"peak_month": 11, "type": "moderate_seasonal","amplitude": 0.20},
    "toner":      {"peak_month": 3,  "type": "mild_seasonal",    "amplitude": 0.12},
    "serum":      {"peak_month": -1, "type": "stable",           "amplitude": 0.08},
    "cleanser":   {"peak_month": -1, "type": "stable",           "amplitude": 0.07},
    "mask":       {"peak_month": 1,  "type": "moderate_seasonal","amplitude": 0.18},
    "oil":        {"peak_month": 10, "type": "moderate_seasonal","amplitude": 0.15},
    "eye":        {"peak_month": -1, "type": "stable",           "amplitude": 0.06},
    "lip":        {"peak_month": 11, "type": "mild_seasonal",    "amplitude": 0.10},
    "general":    {"peak_month": -1, "type": "stable",           "amplitude": 0.10},
}

_TREND_LABELS = {
    0: "strong_decline",
    1: "moderate_decline",
    2: "stable",
    3: "moderate_growth",
    4: "strong_growth",
}


def generate_demand_signals(
    product_id: str,
    product_name: str,
    category: str,
    price: float,
    units_sold: float,
    rng: random.Random,
    seed: int,
) -> Dict:
    """
    Engine 5: Generate hidden demand signals.
    Returns enriched forecast metadata injected into forecast model input.
    """
    import math

    profile = _SEASONALITY_PROFILES.get(category, _SEASONALITY_PROFILES["general"])

    # Trend: seed-driven, biased slightly positive
    trend_bucket = (seed % 7)          # 0-6
    trend_slope = (
         0.020 if trend_bucket == 6 else
         0.012 if trend_bucket == 5 else
         0.005 if trend_bucket >= 3 else
        -0.005 if trend_bucket == 2 else
        -0.012 if trend_bucket == 1 else
        -0.020
    )
    trend_label = (
        "strong_growth"   if trend_slope >  0.015 else
        "moderate_growth" if trend_slope >  0.004 else
        "stable"          if trend_slope >= -0.004 else
        "moderate_decline" if trend_slope >= -0.015 else
        "strong_decline"
    )

    # Seasonality amplitude
    cur_month = __import__("datetime").datetime.now().month - 1  # 0-indexed
    if profile["type"] != "stable" and profile["peak_month"] >= 0:
        dist = abs(cur_month - profile["peak_month"])
        dist = min(dist, 12 - dist)               # wrap December–January
        season_factor = 1.0 + profile["amplitude"] * math.cos(math.pi * dist / 6)
    else:
        season_factor = 1.0 + profile["amplitude"] * rng.uniform(-1, 1)

    # Demand shocks (festivals, promotions) — ~20% of products
    shock_factor = 1.0
    shock_label  = "none"
    if rng.random() < 0.20:
        shock_factor = rng.uniform(1.15, 1.40)
        shock_label  = rng.choice(["festival_boost", "promo_discount",
                                   "influencer_peak", "seasonal_push"])

    # Price sensitivity: higher-priced products are more elastic
    price_sensitivity = round(max(0.1, min(2.0, 25.0 / max(1.0, price))), 3)

    # Effective base units with all signals applied
    effective_units = max(5.0, units_sold * season_factor * shock_factor)

    return {
        "trend_slope":      round(trend_slope, 4),
        "trend_label":      trend_label,
        "season_factor":    round(season_factor, 4),
        "shock_factor":     round(shock_factor, 4),
        "shock_event":      shock_label,
        "price_sensitivity": price_sensitivity,
        "effective_units":  round(effective_units, 1),
        "seasonality_type": profile["type"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 6 — UNIQUENESS ENFORCER
# ─────────────────────────────────────────────────────────────────────────────

def enforce_uniqueness(product_id: str, product_name: str, category: str,
                       price: float) -> Dict:
    """
    Engine 6: Create a stable product fingerprint so no two products
    produce identical ML outputs.
    """
    pid_str = str(product_id)
    h = int(hashlib.md5(pid_str.encode()).hexdigest(), 16)
    product_hash  = round((h % 100000) / 100000.0, 5)

    name_hash     = int(hashlib.md5(product_name.encode()).hexdigest(), 16)
    name_factor   = round((name_hash % 1000) / 1000.0, 3)

    cat_multipliers = {
        "serum": 1.30, "sunscreen": 1.45, "cream": 1.10,
        "cleanser": 1.20, "toner": 1.15, "mask": 0.90,
        "oil": 1.05, "eye": 1.25, "lip": 0.85, "general": 1.00,
    }
    cat_mult = cat_multipliers.get(category, 1.0)

    # Price tier: budget 0.7×, mid 1.0×, premium 1.3×
    price_tier = (
        "budget"  if price < 15  else
        "mid"     if price < 50  else
        "premium"
    )
    price_factor = {"budget": 0.70, "mid": 1.00, "premium": 1.30}[price_tier]

    return {
        "product_hash":       product_hash,
        "name_factor":        name_factor,
        "category_multiplier": cat_mult,
        "price_sensitivity_factor": price_factor,
        "price_tier":         price_tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 7 — SAFETY AUTO-CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

_RISK_SEVERITY: Dict[str, int] = {
    "paraben":         7,
    "fragrance":       4,
    "alcohol":         3,
    "oxybenzone":      8,
    "formaldehyde":    9,
    "sls":             6,
    "benzoyl_peroxide": 5,
    "hydroquinone":    8,
}

_RISK_REASONS: Dict[str, str] = {
    "paraben":         "Paraben-class preservative; potential endocrine disruptor",
    "fragrance":       "Synthetic fragrance; common irritant for sensitive skin",
    "alcohol":         "Denatured alcohol; may impair skin barrier with overuse",
    "oxybenzone":      "Chemical UV filter; potential hormone disruption",
    "formaldehyde":    "Preservative; carcinogenic at elevated concentrations",
    "sls":             "Surfactant; strips natural oils; irritant for sensitive skin",
    "benzoyl_peroxide": "Acne treatment; oxidising agent; can cause dryness",
    "hydroquinone":    "Skin lightener; associated with ochronosis with long-term use",
}

_RISK_INGREDIENT_NAMES: Dict[str, str] = {
    "paraben":         "Paraben Compound",
    "fragrance":       "Fragrance (Parfum)",
    "alcohol":         "Denatured Alcohol",
    "oxybenzone":      "Oxybenzone",
    "formaldehyde":    "Formaldehyde",
    "sls":             "Sodium Lauryl Sulfate",
    "benzoyl_peroxide": "Benzoyl Peroxide",
    "hydroquinone":    "Hydroquinone",
}


def classify_safety(risk_labels: List[str]) -> Dict:
    """
    Engine 7: Compute a safety profile from the generated risk labels.
    Returns harmful_ingredients list, safety_score, status, toxicity_level.
    """
    harmful: List[Dict] = []
    total_penalty = 0

    for label in set(risk_labels):            # deduplicate
        severity = _RISK_SEVERITY.get(label, 5)
        harmful.append({
            "name":     _RISK_INGREDIENT_NAMES.get(label, label.title()),
            "reason":   _RISK_REASONS.get(label, "Potential sensitiser"),
            "severity": severity,
        })
        total_penalty += severity * 5

    safety_score  = max(0.0, float(100 - total_penalty))
    status        = "Safe" if safety_score >= 85 else ("Moderate" if safety_score >= 60 else "Unsafe")
    toxicity_level = round((100.0 - safety_score) / 10.0, 1)

    if not harmful:
        recommendation = "Product is safe. No harmful ingredients detected."
    elif status == "Moderate":
        recommendation = "Use with caution — some potentially harmful ingredients present."
    else:
        recommendation = "Avoid use — contains multiple harmful or restricted ingredients."

    return {
        "safety_score":       safety_score,
        "status":             status,
        "toxicity_level":     toxicity_level,
        "harmful_ingredients": harmful,
        "harmful_count":      len(harmful),
        "recommendation":     recommendation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 8 — SENTIMENT AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_sentiment(reviews: List[str], raw_score: float) -> Dict:
    """
    Engine 8: Build a proper sentiment summary from generated reviews.
    raw_score comes from generate_reviews() — range -1 to +1.
    """
    sentiment = (
        "positive" if raw_score >  0.15 else
        "negative" if raw_score < -0.15 else
        "neutral"
    )
    confidence = round(abs(raw_score), 3)

    positive_count = sum(1 for r in reviews if any(
        w in r.lower() for w in ["love", "amazing", "best", "incredible",
                                  "game changer", "highly recommend", "five stars"]
    ))
    negative_count = sum(1 for r in reviews if any(
        w in r.lower() for w in ["breakout", "irritat", "waste", "disappointed",
                                  "avoid", "redness", "itching"]
    ))
    neutral_count = len(reviews) - positive_count - negative_count

    return {
        "sentiment":       sentiment,
        "score":           round(raw_score, 3),
        "confidence":      confidence,
        "review_count":    len(reviews),
        "positive_count":  positive_count,
        "neutral_count":   max(0, neutral_count),
        "negative_count":  negative_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 9 — FINAL ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def assemble_enriched_product(
    raw: Dict,
    understanding: Dict,
    ingredients_str: str,
    risk_labels: List[str],
    reviews: List[str],
    sentiment_raw: float,
    skin_text: str,
    demand_signals: Dict,
    uniqueness: Dict,
    safety: Dict,
    sentiment_summary: Dict,
) -> Dict:
    """Engine 9: Merge everything into one clean enriched product dict."""
    enriched = dict(raw)

    # Core understanding
    enriched["category"]     = understanding["category"]
    enriched["skin_target"]  = understanding["skin_target"]
    enriched["function"]     = understanding["function"]

    # Ingredients — prefer supplied, fall back to generated
    if not enriched.get("ingredients") or len(str(enriched.get("ingredients", ""))) < 5:
        enriched["ingredients"]           = ingredients_str
        enriched["ingredients_generated"] = True
    else:
        enriched["ingredients_generated"] = False

    enriched["risk_labels"] = risk_labels

    # Reviews
    if not enriched.get("review") or len(str(enriched.get("review", ""))) < 10:
        enriched["review"]           = ". ".join(reviews[:5])
        enriched["all_reviews"]      = reviews
        enriched["reviews_generated"] = True
    else:
        enriched["all_reviews"]       = [enriched["review"]]
        enriched["reviews_generated"] = False

    enriched["has_reviews"] = True

    # Skin text
    if not enriched.get("skin_text"):
        enriched["skin_text"] = skin_text

    # Demand signals
    enriched["demand_signals"] = demand_signals
    enriched["effective_units"] = demand_signals["effective_units"]

    # Uniqueness fingerprint
    enriched.update(uniqueness)

    # Pre-computed ML analysis
    enriched["precomputed_safety"]    = safety
    enriched["precomputed_sentiment"] = sentiment_summary

    # Ensure numeric fields
    enriched.setdefault("price",      25.0)
    enriched.setdefault("units_sold", 100.0)

    enriched["enriched"] = True
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 10 — OUTPUT VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = [
    "product_id", "product_name", "category", "skin_target", "function",
    "ingredients", "review", "has_reviews", "brand",
    "price", "units_sold", "product_hash", "enriched",
    "precomputed_safety", "precomputed_sentiment", "demand_signals",
]

_DEFAULTS: Dict[str, Any] = {
    "category":    "general",
    "skin_target": "normal",
    "function":    "general-care",
    "ingredients": "Aqua (Water), Glycerin, Niacinamide, Panthenol, Allantoin",
    "review":      "A reliable everyday skincare product.",
    "has_reviews": True,
    "brand":       "Unknown",
    "price":       25.0,
    "units_sold":  100.0,
    "product_hash": 0.5,
    "enriched":    True,
    "precomputed_safety":    {"safety_score": 90.0, "status": "Safe",
                              "toxicity_level": 1.0, "harmful_ingredients": [],
                              "harmful_count": 0,
                              "recommendation": "Product is safe."},
    "precomputed_sentiment": {"sentiment": "neutral", "score": 0.0,
                              "confidence": 0.5, "review_count": 1},
    "demand_signals": {"trend_label": "stable", "season_factor": 1.0,
                       "shock_factor": 1.0, "shock_event": "none",
                       "effective_units": 100.0},
}


def validate_output(product: Dict) -> Dict:
    """Engine 10: Guarantee no empty or missing fields."""
    for field in _REQUIRED_FIELDS:
        if field not in product or product[field] is None or product[field] == "":
            product[field] = _DEFAULTS.get(field, "")
            logger.warning("Enrichment: backfilled missing field '%s'", field)

    # Clamp numerics
    product["price"]      = max(0.0, float(product.get("price", 25.0)))
    product["units_sold"] = max(1.0, float(product.get("units_sold", 100.0)))

    return product


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — DataEnricher
# ─────────────────────────────────────────────────────────────────────────────

class DataEnricher:
    """
    Main entry point.  Call DataEnricher.enrich(raw_product_dict) after
    parsing a CSV row.  Returns a fully enriched product dict ready to
    be stored in products.json and consumed by ML endpoints.
    """

    @staticmethod
    def enrich(raw: Dict) -> Dict:
        """
        Transform a minimal product dict into a fully enriched record.

        Parameters
        ----------
        raw : dict
            Must contain at minimum: product_id, product_name.
            Optional (used if present): price, units_sold, ingredients,
            review, brand.

        Returns
        -------
        dict
            Enriched product record with all fields populated.
        """
        product_id   = str(raw.get("product_id", "UNKNOWN"))
        product_name = str(raw.get("product_name", "Unknown Product"))
        price        = float(raw.get("price",      25.0) or 25.0)
        units_sold   = float(raw.get("units_sold", 100.0) or 100.0)

        # Deterministic RNG seeded by product_id so results are stable
        seed    = int(hashlib.md5(product_id.encode()).hexdigest(), 16) % (2**32)
        rng     = random.Random(seed)

        # ── Run all engines ──────────────────────────────────────────────────
        # E1
        understanding = infer_product_understanding(product_name)
        cat    = understanding["category"]
        target = understanding["skin_target"]
        func   = understanding["function"]

        # E2 — mega-dataset aware: pass product_name so keyword boosts apply
        ingredients_str, risk_labels = generate_ingredients(cat, target, func, rng,
                                                            product_name=product_name)

        # E3
        reviews, sentiment_raw = generate_reviews(
            product_name, cat, target, func, risk_labels, rng
        )

        # E4
        skin_text = generate_skin_text(product_name, cat, target, func, rng)

        # E5
        demand_signals = generate_demand_signals(
            product_id, product_name, cat, price, units_sold, rng, seed
        )

        # E6
        uniqueness = enforce_uniqueness(product_id, product_name, cat, price)

        # E7 — run safety through the real ml_service.detect_harmful engine
        # which uses the full 48-keyword harmful dictionary including generic
        # aliases (alcohol, sulphates, etc.) for accurate analysis.
        _final_ing_str = ingredients_str
        if raw.get("ingredients") and len(str(raw.get("ingredients", "")).strip()) > 5:
            _final_ing_str = str(raw["ingredients"])

        try:
            import sys as _sys
            import importlib as _importlib
            _ml = _importlib.import_module("ml_service")
            _svc = getattr(_ml, "svc", None)
            if _svc and hasattr(_svc, "detect_harmful"):
                safety = _svc.detect_harmful(
                    ingredient_text=_final_ing_str,
                    product_id=product_id,
                    product_name=product_name,
                )
                # Remove the 'model_used' and 'product_id/name' keys — only keep ML analysis fields
                safety = {k: v for k, v in safety.items()
                          if k in ("safety_score", "status", "toxicity_level",
                                   "harmful_ingredients", "harmful_count", "recommendation")}
            else:
                safety = classify_safety(risk_labels)
        except Exception as _e:
            logger.warning("ml_service.detect_harmful unavailable in enricher: %s", _e)
            safety = classify_safety(risk_labels)

        # E8
        sentiment_summary = aggregate_sentiment(reviews, sentiment_raw)

        # E9
        enriched = assemble_enriched_product(
            raw, understanding, ingredients_str, risk_labels,
            reviews, sentiment_raw, skin_text, demand_signals,
            uniqueness, safety, sentiment_summary,
        )

        # E10
        enriched = validate_output(enriched)

        logger.info(
            "Enriched '%s' | cat=%s target=%s risks=%s safety=%s sentiment=%s",
            product_name[:40],
            cat, target,
            risk_labels or "none",
            safety["status"],
            sentiment_summary["sentiment"],
        )
        return enriched

    @staticmethod
    def enrich_batch(products: List[Dict]) -> List[Dict]:
        """Enrich a list of products, returning the enriched list."""
        return [DataEnricher.enrich(p) for p in products]
