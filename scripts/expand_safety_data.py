import pickle
import os
from pathlib import Path

def expand_safety_data():
    # Base directory
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Comprehensive Ingredient Dictionary
    # Severity: 1-10 (1-3 Low/Safe, 4-7 Medium, 8-10 High)
    data = {
        # --- HIGH RISK / HARMFUL ---
        "methylparaben": {"name": "Methylparaben", "severity": 8, "reason": "Potential endocrine disruptor; associated with breast cancer risk."},
        "propylparaben": {"name": "Propylparaben", "severity": 9, "reason": "High-risk endocrine disruptor; mimics estrogen."},
        "butylparaben": {"name": "Butylparaben", "severity": 9, "reason": "Restricted paraben; strong endocrine disrupting effects."},
        "ethylparaben": {"name": "Ethylparaben", "severity": 8, "reason": "Preservative with endocrine disrupting potential."},
        "formaldehyde": {"name": "Formaldehyde", "severity": 10, "reason": "Known human carcinogen; severe respiratory and skin irritant."},
        "quaternium-15": {"name": "Quaternium-15", "severity": 9, "reason": "Formaldehyde-releaser; common allergen and carcinogen risk."},
        "dmdm hydantoin": {"name": "DMDM Hydantoin", "severity": 8, "reason": "Formaldehyde-releaser; skin and lung irritant."},
        "triclosan": {"name": "Triclosan", "severity": 9, "reason": "Endocrine disruptor; impacts thyroid function and gut health."},
        "hydroquinone": {"name": "Hydroquinone", "severity": 10, "reason": "Skin bleaching agent linked to organ toxicity and skin thinning."},
        "toluene": {"name": "Toluene", "severity": 10, "reason": "Potent neurotoxin; can impact fetal development and organ health."},
        "lead acetate": {"name": "Lead Acetate", "severity": 10, "reason": "Heavy metal; neurotoxin and developmental hazard."},
        "coal tar": {"name": "Coal Tar", "severity": 10, "reason": "By-product of coal; known human carcinogen."},
        "oxybenzone": {"name": "Oxybenzone", "severity": 8, "reason": "Chemical UV filter; high rate of skin allergy and hormone disruption."},
        "phthalates": {"name": "Phthalates", "severity": 9, "reason": "Endocrine disruptors linked to reproductive issues."},
        "dibutyl phthalate": {"name": "Dibutyl Phthalate (DBP)", "severity": 10, "reason": "Developmental and reproductive toxin."},
        "diethyl phthalate": {"name": "Diethyl Phthalate (DEP)", "severity": 8, "reason": "Possible endocrine disruptor."},

        # --- MEDIUM RISK / MODERATE ---
        "sodium lauryl sulfate": {"name": "Sodium Lauryl Sulfate (SLS)", "severity": 6, "reason": "Harsh surfactant; strips skin of natural oils and causes irritation."},
        "sls": {"name": "SLS", "severity": 6, "reason": "Harsh surfactant; known skin irritant."},
        "sodium laureth sulfate": {"name": "Sodium Laureth Sulfate (SLES)", "severity": 5, "reason": "Slightly milder than SLS but can be contaminated with carcinogens."},
        "sles": {"name": "SLES", "severity": 5, "reason": "Ethoxylated surfactant; risk of 1,4-dioxane contamination."},
        "fragrance": {"name": "Fragrance/Parfum", "severity": 6, "reason": "Umbrella term for thousands of chemicals; high risk of allergy and irritation."},
        "parfum": {"name": "Parfum", "severity": 6, "reason": "May contain hidden allergens and phthalates."},
        "phenoxyethanol": {"name": "Phenoxyethanol", "severity": 5, "reason": "Preservative that can be a skin irritant at concentrations above 1%."},
        "alcohol denat": {"name": "Denatured Alcohol", "severity": 5, "reason": "Drying alcohol; can compromise the skin barrier over time."},
        "isopropyl alcohol": {"name": "Isopropyl Alcohol", "severity": 6, "reason": "Highly drying and irritating for sensitive skin."},
        "ethanol": {"name": "Ethanol", "severity": 4, "reason": "Drying effect; can enhance penetration of other chemicals."},
        "avobenzone": {"name": "Avobenzone", "severity": 4, "reason": "Chemical sun filter; relatively safe but can cause contact dermatitis."},
        "mineral oil": {"name": "Mineral Oil", "severity": 4, "reason": "Petroleum derivative; can clog pores and trap bacteria (comedogenic)."},
        "paraffin": {"name": "Paraffin", "severity": 4, "reason": "Derived from petroleum; occlusive and may block skin respiration."},
        "petrolatum": {"name": "Petrolatum", "severity": 4, "reason": "Safe when pure, but potentially contaminated with polycyclic aromatic hydrocarbons."},
        "synthetic colors": {"name": "Synthetic Colors (CI/FD&C)", "severity": 5, "reason": "Coal tar derivatives; may cause heavy metal contamination."},

        # --- LOW RISK / SAFE & BENEFICIAL ---
        "water": {"name": "Aqua/Water", "severity": 1, "reason": "Primary solvent; safe and essential for hydration."},
        "aqua": {"name": "Aqua", "severity": 1, "reason": "Safe solvent."},
        "hyaluronic acid": {"name": "Hyaluronic Acid", "severity": 1, "reason": "Powerful humectant; naturally occurring in skin, boosts hydration."},
        "sodium hyaluronate": {"name": "Sodium Hyaluronate", "severity": 1, "reason": "Salt form of Hyaluronic Acid; penetrates deep into skin."},
        "niacinamide": {"name": "Niacinamide (Vitamin B3)", "severity": 1, "reason": "Highly beneficial; reduces inflammation, pores, and improves barrier."},
        "glycerin": {"name": "Glycerin", "severity": 1, "reason": "Classic humectant; safe and effective for all skin types."},
        "panthenol": {"name": "Panthenol (Vitamin B5)", "severity": 1, "reason": "Soothing and moisturizing agent; promotes skin healing."},
        "ceramides": {"name": "Ceramides", "severity": 1, "reason": "Essential lipids that restore and protect the skin barrier."},
        "ceramide np": {"name": "Ceramide NP", "severity": 1, "reason": "Helps maintain the skin's moisture balance."},
        "squalane": {"name": "Squalane", "severity": 1, "reason": "Biocompatible lipid; mimics skin's natural oils for deep hydration."},
        "allantoin": {"name": "Allantoin", "severity": 1, "reason": "Soothing and anti-irritant; derived from comfrey plant."},
        "tocopherol": {"name": "Tocopherol (Vitamin E)", "severity": 2, "reason": "Potent antioxidant; protects skin from environmental damage."},
        "vitamin c": {"name": "Vitamin C", "severity": 1, "reason": "Antioxidant; brightens skin and boosts collagen production."},
        "ascorbic acid": {"name": "Ascorbic Acid", "severity": 1, "reason": "Purest form of Vitamin C."},
        "green tea extract": {"name": "Green Tea Extract", "severity": 1, "reason": "Soothing antioxidant; reduces redness and inflammation."},
        "aloe barbadensis": {"name": "Aloe Vera", "severity": 1, "reason": "Soothing and hydrating botanical extract."},
        "jojoba oil": {"name": "Jojoba Oil", "severity": 1, "reason": "Safe botanical oil; very similar to skin's natural sebum."},
        "shea butter": {"name": "Shea Butter", "severity": 1, "reason": "Rich emollient; safe for dry and sensitive skin."},
        "zinc oxide": {"name": "Zinc Oxide", "severity": 1, "reason": "Mineral UV filter; extremely safe and soothing for sensitive skin."},
        "titanium dioxide": {"name": "Titanium Dioxide", "severity": 1, "reason": "Mineral sunscreen; non-irritating and safe."},
        "stearic acid": {"name": "Stearic Acid", "severity": 2, "reason": "Safe fatty acid used as an emollient and emulsifier."},
        "glyceryl stearate": {"name": "Glyceryl Stearate", "severity": 2, "reason": "Common, safe emulsifier for creamy textures."},
        "butylene glycol": {"name": "Butylene Glycol", "severity": 2, "reason": "Common humectant and solvent; generally considered safe."},
    }
    
    # Add parent keywords to handle variations
    data["parabens"] = {"name": "Parabens", "severity": 9, "reason": "Group of preservatives linked to hormone disruption."}
    data["sulphates"] = {"name": "Sulphates", "severity": 6, "reason": "Harsh surfactants."}
    data["sulfates"] = {"name": "Sulfates", "severity": 6, "reason": "Harsh surfactants."}
    
    payload = {
        "harmful_keywords": data,
        "metadata": {
            "version": "2.0.0",
            "count": len(data),
            "last_updated": "2024-05-07"
        }
    }
    
    with open(models_dir / "harmful_detector.pkl", "wb") as f:
        pickle.dump(payload, f)
    
    print(f"Success! Harmful detector updated with {len(data)} ingredients.")

if __name__ == "__main__":
    expand_safety_data()
