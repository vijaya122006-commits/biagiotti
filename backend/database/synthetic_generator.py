"""
database/synthetic_generator.py — Generates realistic sandbox data for new dealers.
Called during onboarding when mode = "sandbox".
"""
from __future__ import annotations
import hashlib
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database.models import db, Product, Inventory, Sale, Review, Notification, AnalysisResult

logger = logging.getLogger("synthetic_generator")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ─── Indian market brand names ─────────────────────────────────────────────────

INDIAN_BRANDS = [
    "Lakme", "Mamaearth", "Biotique", "Himalaya", "Plum", "Dot & Key",
    "Minimalist", "WOW Skin Science", "Nykaa Cosmetics", "Forest Essentials",
    "Kama Ayurveda", "Re'equil", "Cetaphil", "Neutrogena", "L'Oreal Paris",
    "Garnier", "VLCC", "Lotus Herbals", "MCaffeine", "Sugarboo",
]

INDIAN_FIRST_NAMES = [
    "Priya", "Ananya", "Ritu", "Sunita", "Kavya", "Meera", "Pooja", "Shreya",
    "Divya", "Neha", "Rahul", "Arjun", "Vikram", "Suresh", "Rajesh", "Arun",
    "Kiran", "Deepa", "Nidhi", "Swati", "Ankita", "Pallavi", "Sneha", "Bhavna",
    "Ritika", "Sonal", "Aarti", "Puja", "Riya", "Tanya", "Varun", "Amit",
]

INDIAN_LAST_NAMES = [
    "Sharma", "Kumar", "Singh", "Verma", "Gupta", "Patel", "Mehta", "Iyer",
    "Reddy", "Nair", "Joshi", "Malhotra", "Kapoor", "Aggarwal", "Bose",
    "Mukherjee", "Rao", "Pillai", "Menon", "Srinivasan", "Banerjee", "Das",
]

CATEGORY_DISTRIBUTION = {
    "Moisturizer": 100,
    "Serum": 90,
    "Sunscreen": 60,
    "Foundation": 50,
    "Cleanser": 50,
    "Toner": 40,
    "Eye Cream": 35,
    "Lip Care": 25,
    "Mask": 25,
    "Hair Care": 25,
}

PRICE_TIERS = [
    (0.40, (99, 499)),     # Budget 40%
    (0.35, (500, 1499)),   # Mid 35%
    (0.20, (1500, 3999)),  # Premium 20%
    (0.05, (4000, 12000)), # Luxury 5%
]

INGREDIENT_MAP = {
    "Moisturizer":  "Aqua, Glycerin, Hyaluronic Acid, Ceramide NP, Niacinamide, Panthenol, Sodium PCA",
    "Serum":        "Ascorbic Acid, Niacinamide, Hyaluronic Acid, Glycolic Acid, Retinol, Peptides",
    "Sunscreen":    "Zinc Oxide, Titanium Dioxide, Avobenzone, Homosalate, Octisalate, Aloe Vera",
    "Foundation":   "Talc, Titanium Dioxide, Iron Oxides, Silica, Dimethicone, Mica, Glycerin",
    "Cleanser":     "Sodium Laureth Sulfate, Cocamidopropyl Betaine, Salicylic Acid, Aloe, Glycerin",
    "Toner":        "Aqua, Centella Asiatica, Niacinamide, Witch Hazel, Glycerin, Hyaluronic Acid",
    "Eye Cream":    "Retinol, Caffeine, Peptides, Hyaluronic Acid, Vitamin K, Ceramide",
    "Lip Care":     "Shea Butter, Beeswax, Vitamin E, Jojoba Oil, Aloe Vera, Castor Oil",
    "Mask":         "Kaolin Clay, Bentonite, Salicylic Acid, Tea Tree Oil, Glycerin",
    "Hair Care":    "Keratin, Argan Oil, Biotin, Panthenol, Cetrimonium Chloride, Glycerin",
}

SEASONAL_MULTIPLIERS = {
    1: 0.70,   # January — post-Diwali dip
    2: 1.60,   # February — Valentine's
    3: 1.80,   # March — Summer/Wedding start
    4: 1.80,   # April — Summer
    5: 1.60,   # May — Wedding season
    6: 1.40,   # June
    7: 0.85,   # July — Monsoon
    8: 0.85,   # August — Monsoon
    9: 1.20,   # September — pre-festive
    10: 2.20,  # October — Diwali
    11: 2.20,  # November — Diwali
    12: 1.30,  # December — Christmas/year-end
}

REVIEW_TEMPLATES = {
    5: {
        "oily": [
            "Controls oil throughout the day, doesn't leave any greasy residue. Highly recommend!",
            "My T-zone stays matte for hours after using this product.",
            "No breakouts since I started using this, totally non-comedogenic.",
            "Finally found something that controls shine without drying my skin out.",
            "Game changer for oily skin types! Light, non-sticky formula.",
            "Perfect for the humid Indian summers — no midday shine at all!",
            "Used this through a Chennai summer and my face stayed shine-free. Impressed!",
            "Lightweight texture, absorbs quickly and leaves a matte finish. Love it.",
            "My oily skin has finally found its match. No more blotting papers needed!",
            "Pores look visibly smaller after consistent use. Absolutely brilliant.",
            "Oil-free formula lives up to its name. My skin feels clean and fresh all day.",
            "Even in 40°C weather my face stays matte — that's saying something!",
            "The sebum control is real! No patchiness either, just smooth skin.",
            "Best product for my combo-oily skin. The T-zone stays controlled for 8+ hours.",
            "Finally a product that doesn't slide off my face by afternoon. Game changer!",
            "Lightweight but effective. My skin feels breathable and not suffocated.",
            "Controls oil without stripping moisture — a rare balance I've been searching for.",
            "No more midday touch-ups! This product genuinely keeps shine at bay.",
            "Used through monsoon season and it still controlled oil better than anything I've tried.",
            "Non-greasy, fast-absorbing, and the oil control is outstanding. Will repurchase!",
        ],
        "dry": [
            "Finally something that keeps my skin hydrated for 12+ hours!",
            "No more flakiness, skin feels plump and moisturized all day.",
            "Best moisturizer I've tried. Works perfectly for my dry skin.",
            "Rich but not greasy. My skin drinks it up instantly.",
            "Transformed my dry, dull skin into something glowing and healthy.",
            "Winter skin saviour! No more tight, itchy feeling after washing my face.",
            "Intensely hydrating without feeling heavy. Perfect for dry skin like mine.",
            "Skin feels like it has its own glow after just one week of use!",
            "My dry patches completely disappeared after 10 days. Remarkable results.",
            "Wakes up looking dewy and plump — no more dull, dry mornings.",
            "This is the only product that has kept my extremely dry skin comfortable all day.",
            "The hydration literally lasts overnight. Skin feels amazing in the morning.",
            "Thick enough for deep hydration but absorbs without leaving a residue. Perfect.",
            "My flaky skin is a thing of the past! Two weeks in and the difference is stunning.",
            "Deeply moisturizing formula that my dry skin absolutely loves. Worth every penny!",
            "Even on the driest parts of my face this product delivers visible hydration.",
            "Skin is plump and supple within minutes of application. Truly impressive.",
            "This has completely replaced my previous moisturizer — it's simply better in every way.",
            "Works beautifully under makeup too! No pilling and the hydration lasts all day.",
            "Finally found my holy grail moisturizer after years of searching. Dry skin rejoice!",
        ],
        "sensitive": [
            "Gentle formula, no redness or irritation even on my reactive skin.",
            "Fragrance-free and dermatologist tested, exactly what my skin needed.",
            "No stinging, no breakouts. Safe for even the most sensitive skin.",
            "Used this for 2 months, zero irritation. Brilliant formulation.",
            "My rosacea-prone skin finally has a product it tolerates beautifully.",
            "No burning, no tingling — just calm, comfortable skin. Incredible!",
            "After years of reactions to skincare, this is the first product I've trusted fully.",
            "The soothing effect is almost immediate. Perfect for my easily inflamed skin.",
            "Hypoallergenic and genuinely gentle. Not just a marketing claim — it actually works.",
            "My dermatologist recommended this and it has been a lifesaver for my sensitive skin.",
            "No redness, no flare-ups, no irritation. This is truly gentle enough for reactive skin.",
            "My skin calmed down significantly within the first few uses. Truly effective.",
            "Wore this through a stressful week and my skin remained calm and clear. Love it.",
            "Perfect for eczema-prone skin — no triggers and great hydration.",
            "The minimalist ingredient list works in its favour. No nonsense, just results.",
            "My usual culprit ingredients are absent here and my skin thanks me daily.",
            "First product in years that hasn't caused a reaction on my ultra-sensitive skin.",
            "Lightweight and soothing — exactly what sensitive skin needs without compromise.",
            "Even during my worst skin flare-ups this product didn't aggravate things further.",
            "Tested this for 6 weeks — zero reactions, only improvements. Fully recommended.",
        ],
        "all": [
            "Absolutely love this product! Results visible within 2 weeks.",
            "Best purchase I've made in skincare. Worth every rupee!",
            "Exceeded all my expectations. Skin looks 5 years younger.",
            "My entire family uses this now. Incredible product!",
            "Highly recommend to anyone who wants to upgrade their skincare routine.",
            "Obsessed with this! Have already repurchased twice.",
            "Delivers exactly what it promises and then some. Outstanding quality.",
            "Results speak for themselves — my skin has never looked better.",
            "This product has completely transformed my skincare routine for the better.",
            "Could not be happier with this purchase. Will be a staple forever.",
            "Genuinely the best skincare investment I've made this year.",
            "Every single claim on the label is backed up by real results. Bravo!",
            "My skin glows differently since I started using this. Can't imagine going without it.",
            "Premium formulation that actually delivers. Worth the splurge!",
            "This product has earned its permanent spot on my bathroom shelf.",
            "Tried many alternatives but nothing compares to this one.",
            "Simple to use, impressive results, great value. What more could you want?",
            "After just one month, friends have been asking what my skincare secret is!",
            "The texture, the scent, the results — everything about this is 10/10.",
            "Completely changed how my skin feels. I feel more confident than ever.",
            "My skin barrier feels so healthy and strong since I started using this.",
            "Results that justify every single rupee spent. Absolutely worth it.",
            "This is one of those rare products where the results are actually visible in photos.",
            "Repurchased three times already — that says everything!",
            "Introduced this to my skincare group and everyone ordered within the week.",
        ],
    },
    "category_specific": {
        "Foundation": [
            "The shade match is perfect for my skin tone. Blends like a dream!",
            "Great coverage without feeling cakey. Lasts through my whole shift.",
            "Doesn't oxidize or turn orange. Best foundation I've found in India.",
            "Really natural finish, looks like my skin but better.",
            "Good buildable coverage. Covers my redness easily.",
            "Works well even without a primer. Highly impressed with the formula.",
        ],
        "Serum": [
            "Lightweight consistency, absorbs instantly. Seeing a glow already!",
            "Been using this for a week and my dark spots are starting to fade.",
            "Perfect addition to my night routine. Not sticky at all.",
            "My skin feels so much smoother since I started using this serum.",
            "Very concentrated formula, a little goes a long way.",
            "I love that it's fragrance-free and potent. High quality serum.",
        ],
        "Sunscreen": [
            "Zero white cast! Finally a sunscreen that works on deeper skin tones.",
            "Non-greasy finish, doesn't make my face look like a disco ball.",
            "Great under makeup, doesn't pill. Essential for my daily routine.",
            "The broad spectrum protection is real. No tanning even after beach days.",
            "Soothing formula, doesn't sting my eyes like other sunscreens do.",
            "Absorbs quickly and leaves a nice matte-dewy balance.",
        ],
        "Moisturizer": [
            "Keeps my skin hydrated all day without feeling heavy or oily.",
            "The texture is amazing — feels like a luxury product for a great price.",
            "My skin barrier has never felt stronger. Perfect for everyday use.",
            "A little goes a long way. Deeply nourishing and very effective.",
            "Absorbs beautifully and creates a smooth canvas for makeup.",
            "Finally found a moisturizer that doesn't cause breakouts.",
        ],
        "Cleanser": [
            "Gentle but effective. Removes all my makeup without stripping my skin.",
            "My face feels clean and refreshed, not tight or dry at all.",
            "Love the subtle scent and how it foams up just enough.",
            "Best cleanser I've used for my sensitive skin. Very soothing.",
            "Doesn't leave any residue. Perfectly preps my skin for the rest of my routine.",
            "Genuinely helps with my acne. Skin looks clearer after just two weeks.",
        ],
        "Toner": [
            "So refreshing! My pores look visibly smaller and skin feels balanced.",
            "Preps my skin perfectly for serums. Love the hydrating feel.",
            "Not drying at all, which is rare for a toner. Very impressed.",
            "Helps calm my redness instantly. A staple in my routine now.",
            "The consistency is just right. Doesn't feel like just water.",
            "Great for evening out skin texture. My face feels so smooth.",
        ],
    },
    4: [
        "Really good product overall, just wish it came with a better pump dispenser.",
        "Works as described, though it took about 3 weeks to see noticeable results.",
        "Great for everyday use. Only wish the tube size was bigger for the price.",
        "Effective formula, love the ingredient list. Packaging could use an upgrade.",
        "Skin has improved noticeably. Took off one star because fragrance is a bit present.",
        "Good daily moisturizer, non-comedogenic and lightweight. Happy with it.",
        "Love the texture but wish it was available in a larger size at a similar price point.",
        "Works better than expected. The only downside is slow availability on restocking.",
        "Solid product with visible improvements in skin texture after 2 weeks of use.",
        "Nice formulation but the travel-size packaging is not very practical.",
        "Good performance but I expected faster visible results based on the claims.",
        "Really good for the price. Not revolutionary but consistently effective.",
        "Happy with the results, though skin took a few days to adjust to the formula.",
        "Works beautifully in cooler months. Not quite as effective during humid summers.",
        "Smells faintly pleasant without being overpowering. Performance is solid.",
        "Four stars because the product works great but the cap leaks occasionally.",
        "Good formulation that actually does what the label says. Worth trying.",
        "Improvement in my skin has been gradual but definitely real. Recommend it.",
        "I like how lightweight this is without sacrificing effectiveness. Good buy.",
        "Solid choice in its category. Needs minor improvements in the dispenser design.",
        "Works well consistently. Just not spectacular enough for five stars for me personally.",
        "Good value for money. Skin feels better and looks cleaner since I started using it.",
        "Effective but the texture is slightly thicker than I prefer for everyday use.",
        "Nice results after a month of use. Dropped a star for the strong scent initially.",
        "Would definitely recommend with the note that it takes patience to see full results.",
    ],
    3: [
        "Does what it claims but packaging could be better.",
        "Decent product for the price, nothing extraordinary.",
        "Works okay for normal skin, might not suit everyone.",
        "Average performance. There are better alternatives out there.",
        "Product is fine but fragrance is a bit strong for daily use.",
        "Skin felt okay but I didn't see the dramatic results I was hoping for.",
        "The product works but I'm unsure if I'd repurchase at full price.",
        "Middling results — skin feels neither better nor worse after a month.",
        "Not bad, not great. Just an average addition to my routine.",
        "I expected more given the price point. The results are underwhelming.",
        "Does a basic job of hydrating but nothing that sets it apart from cheaper options.",
        "Product is functional but the scent is synthetic and lingers a bit too long.",
        "Usable product but I've seen better results from less expensive alternatives.",
        "The texture was unexpected — different from what I imagined based on the description.",
        "It's okay for what it is, but I think there are better options at a similar price.",
        "Neutral experience. Skin feels okay but I'm not rushing to repurchase.",
        "Three stars because it isn't bad, but it isn't exciting or impressive either.",
        "Serves its purpose without any real issues, but also without any real excitement.",
        "Got the job done but I expected more from a product in this price range.",
        "Texture is a bit heavy for my liking and it takes too long to absorb.",
        "Not the worst but the results were inconsistent across different weeks.",
        "The product seems decent but results took longer than advertised to show up.",
        "Average daily moisturizer — does the job but nothing beyond the basics.",
        "Mild improvement in skin feel but not enough to justify the premium price tag.",
        "Fine product, nothing to rave about. Would consider it if on a discount.",
    ],
    2: [
        "Underwhelming results after a month of consistent use. Not worth the money.",
        "The formula feels cheap despite the premium pricing. Disappointed.",
        "Didn't see any improvement and it made my skin slightly oilier than before.",
        "Not for me — the texture is too thick and it pills under makeup.",
        "Very disappointing. Expected better given the brand's reputation.",
        "After 3 weeks I see zero difference in my skin. Waste of money.",
        "The product oxidises quickly after opening. Not acceptable at this price.",
        "Left a white cast on my skin that wouldn't go away for hours.",
        "Clogs pores despite claiming to be non-comedogenic. Broke out within 2 weeks.",
        "The ingredients list is impressive but the results don't match. Overhyped.",
        "Packaging is great, product is not. Goes rancid-smelling too quickly.",
        "Makes skin look dull and grey rather than radiant as claimed.",
        "Way too sticky. Still greasy 40 minutes after application.",
        "Completely broke out my normally clear skin. Very disappointed.",
        "The smell is pleasant but the actual performance is below average.",
        "Burned slightly on application which is not something I can tolerate.",
        "Left my skin feeling tight and uncomfortable rather than soft and nourished.",
        "Two months in and my skin condition has actually worsened since I started using this.",
        "Gave it 6 weeks and results are marginal at best. Not recommending.",
        "For this price I expected a transformative experience. Instead, nothing.",
    ],
    1: [
        "Broke out badly after 3 days of use, returning immediately.",
        "Strong fragrance caused irritation on my sensitive skin.",
        "Terrible consistency, product separated after 2 weeks.",
        "Completely ineffective, saw zero results after 4 weeks.",
        "Wasted money. Packaging leaked and product smells odd.",
        "Severe allergic reaction — red bumps all over my face within 24 hours.",
        "Caused chemical burn sensation on my cheeks. Returned and never looking back.",
        "Made my skin peel. Contacted the brand and they were unhelpful. Avoid.",
        "Worst skincare product I have ever used. Made everything worse.",
        "Smells rancid right out of the box. Product was clearly poorly stored.",
        "Gave me contact dermatitis. Had to see a dermatologist to recover.",
        "Completely ruined my skin barrier. Took months to recover after stopping use.",
        "Misleading claims. Nothing on the label is actually reflected in real-world results.",
        "Product separated into water and oil immediately. Clearly a quality issue.",
        "Zero efficacy and my skin broke out like never before. Absolutely avoid.",
        "Felt burning immediately upon application. Had to rinse off. Dangerous!",
        "Made my dark spots worse, not better. Money and time completely wasted.",
        "Texture was completely wrong — clumpy and impossible to blend evenly.",
        "Caused stinging and left a rash that took two weeks to heal. Terrible product.",
        "The product literally changed colour in the tube within one week. Expired?",
    ],
}

# Simple namespace to track used review bodies per product during generation
class _generate_reviews_state:
    """Dynamically stores {product_id: set_of_used_bodies} as attributes."""
    pass


def _random_name():
    return f"{random.choice(INDIAN_FIRST_NAMES)} {random.choice(INDIAN_LAST_NAMES)}"


def _random_price():
    """Pick a price in the Indian market distribution."""
    r = random.random()
    cumulative = 0.0
    for pct, (lo, hi) in PRICE_TIERS:
        cumulative += pct
        if r <= cumulative:
            return round(random.uniform(lo, hi), -1)
    return 499.0


def _skin_from_ingredients(ingredients: str) -> str:
    ing = ingredients.lower()
    if 'salicylic' in ing or 'niacinamide' in ing:
        return 'oily, acne'
    if 'retinol' in ing:
        return 'dry, normal'
    if 'aloe' in ing or 'centella' in ing:
        return 'sensitive, all'
    if 'hyaluronic' in ing or 'ceramide' in ing:
        return 'dry, normal'
    return 'combination, normal'


def _generate_monthly_sales(product_id: str, base_units: float, trend: str, months: int = 12) -> list:
    """Generate monthly sales with Indian seasonal patterns."""
    seed = int(hashlib.md5(str(product_id).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    now = datetime.utcnow()

    sales = []
    for i in range(months):
        month_offset = months - 1 - i
        dt = now - timedelta(days=30 * month_offset)
        month = dt.month
        seasonal = SEASONAL_MULTIPLIERS.get(month, 1.0)

        if trend == 'surge':
            trend_factor = 1.0 + (i * random.uniform(0.05, 0.15))
        elif trend == 'decline':
            trend_factor = 1.0 - (i * random.uniform(0.03, 0.08))
            trend_factor = max(0.2, trend_factor)
        else:
            trend_factor = 1.0 + rng.uniform(-0.02, 0.02)

        val = base_units * seasonal * trend_factor
        noise = rng.normal(0, val * 0.1)
        val = max(1.0, round(val + noise, 1))
        sales.append((dt.year, dt.month, val))

    return sales


def generate_synthetic_data_for_dealer(dealer_id: int):
    """
    Generate 500 synthetic products + 12 months of sales + reviews
    + inventory for a sandbox dealer.
    """
    logger.info("Generating synthetic data for dealer %d...", dealer_id)

    # Reset per-product review body tracking state
    for attr in list(vars(_generate_reviews_state)):
        if not attr.startswith('__'):
            delattr(_generate_reviews_state, attr)

    # Pre-assign problem categories
    product_counter = [0]
    understock_targets = set(range(0, 40))
    overstock_targets = set(range(40, 65))
    harmful_high_targets = set(range(65, 73))
    harmful_medium_targets = set(range(73, 88))
    decline_targets = set(range(88, 100))
    surge_targets = set(range(100, 120))

    harmful_high_ingredients = "Methylparaben, Mercury Chloride, Lead Acetate, Formaldehyde"
    harmful_medium_ingredients = "Methylparaben, Phenoxyethanol, Sodium Laureth Sulfate"

    all_products = []
    all_inventory = []
    all_sales = []
    all_reviews = []

    # Build product list from category distribution
    product_specs = []
    for cat, count in CATEGORY_DISTRIBUTION.items():
        for _ in range(count):
            product_specs.append(cat)

    random.shuffle(product_specs)

    for idx, category in enumerate(product_specs):
        n = product_counter[0]
        product_counter[0] += 1

        brand = random.choice(INDIAN_BRANDS)
        price = _random_price()
        ingredients = INGREDIENT_MAP.get(category, "Aqua, Glycerin, Extracts")

        # Override ingredients for harmful products
        if n in harmful_high_targets:
            ingredients = harmful_high_ingredients
        elif n in harmful_medium_targets:
            ingredients = harmful_medium_ingredients

        product_name = f"{brand} {category} {'Pro' if price > 1500 else 'Daily'} {n:03d}"
        product_id = f"SYN_{dealer_id}_{n:04d}"
        skin = _skin_from_ingredients(ingredients)

        product = Product(
            product_id=product_id,
            dealer_id=dealer_id,
            product_name=product_name,
            brand=brand,
            category=category,
            price=price,
            cost_price=round(price * 0.40, 2),
            ingredients=ingredients,
            skin_suitability=skin,
            is_verified=False,
        )
        all_products.append(product)

        # ── Sales trend assignment ──────────────────────────────────────────
        if n in surge_targets:
            trend = 'surge'
        elif n in decline_targets:
            trend = 'decline'
        else:
            trend = 'stable'

        base_units = max(10.0, price / 10.0 * random.uniform(0.5, 2.0))
        monthly = _generate_monthly_sales(product_id, base_units, trend)

        for year, month, units in monthly:
            sale_date = datetime(year, month, 1)
            all_sales.append(Sale(
                product_id=product_id,
                dealer_id=dealer_id,
                brand=brand,
                region="South India",
                city=random.choice(["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Delhi", "Pune"]),
                event_type="retail",
                year=year,
                month=month,
                units_sold=units,
                revenue=round(units * price, 2),
                sale_date=sale_date,
            ))

        # ── Inventory ─────────────────────────────────────────────────────
        avg_monthly = sum(u for _, _, u in monthly) / len(monthly)
        daily_rate = avg_monthly / 30.0

        if n in understock_targets:
            # 1-6 days of stock
            days = random.randint(1, 6)
            stock = max(1.0, round(daily_rate * days, 0))
        elif n in overstock_targets:
            # 120-200 days of stock
            days = random.randint(120, 200)
            stock = round(daily_rate * days, 0)
        else:
            days = random.randint(20, 90)
            stock = round(daily_rate * days, 0)

        all_inventory.append(Inventory(
            product_id=product_id,
            dealer_id=dealer_id,
            current_stock=max(1.0, stock),
            reorder_level=round(daily_rate * 14, 0),
            lead_time_days=random.randint(7, 21),
            last_restocked=datetime.utcnow() - timedelta(days=random.randint(1, 60)),
        ))

        # ── Reviews (5–15 per product) ────────────────────────────────────
        if price > 2000:
            quality = 'high'
            target_avg = random.uniform(4.2, 4.8)
        elif price > 800:
            quality = 'medium'
            target_avg = random.uniform(3.5, 4.1)
        else:
            quality = 'low'
            target_avg = random.uniform(2.5, 3.4)

        num_reviews = random.randint(5, 15)
        for _ in range(num_reviews):
            # Pick rating based on quality tier
            r = random.random()
            if quality == 'high':
                if r < 0.60:
                    rating = 5
                elif r < 0.85:
                    rating = 4
                elif r < 0.95:
                    rating = 3
                else:
                    rating = random.choice([1, 2])
            elif quality == 'medium':
                if r < 0.35:
                    rating = 5
                elif r < 0.65:
                    rating = 4
                elif r < 0.85:
                    rating = 3
                else:
                    rating = random.choice([1, 2])
            else:
                if r < 0.20:
                    rating = 5
                elif r < 0.40:
                    rating = 4
                elif r < 0.65:
                    rating = 3
                else:
                    rating = random.choice([1, 2])

            # Pick review body — unique per product
            primary_skin = skin.split(',')[0].strip() if skin else 'all'
            used_bodies = getattr(_generate_reviews_state, product_id, set())
            if rating == 5:
                skin_templates = REVIEW_TEMPLATES[5].get(primary_skin, REVIEW_TEMPLATES[5]['all'])
                pool = skin_templates + REVIEW_TEMPLATES[5]['all']
            elif rating >= 4:
                pool = REVIEW_TEMPLATES[4]
            elif rating >= 3:
                pool = REVIEW_TEMPLATES[3]
            elif rating >= 2:
                pool = REVIEW_TEMPLATES[2]
            else:
                pool = REVIEW_TEMPLATES[1]
            
            # --- CATEGORY AWARENESS ---
            # If the product has a category, try to inject category-specific reviews for high ratings
            if rating >= 4:
                cat_pool = REVIEW_TEMPLATES.get("category_specific", {}).get(product.category, [])
                if cat_pool:
                    # 40% chance to use a category-specific review if available
                    if random.random() < 0.4:
                        pool = cat_pool
            
            # Try to pick unique body for this product
            body = random.choice(pool)
            for _attempt in range(20):
                candidate = random.choice(pool)
                if candidate not in used_bodies:
                    body = candidate
                    break
            used_bodies.add(body)
            setattr(_generate_reviews_state, product_id, used_bodies)

            # Compute sentiment
            sent_score = 0.0
            for pos in ["love", "amazing", "best", "perfect", "recommend"]:
                if pos in body.lower():
                    sent_score += 0.3
            for neg in ["broke out", "irritation", "terrible", "zero results", "awful"]:
                if neg in body.lower():
                    sent_score -= 0.4
            sent_score = max(-1.0, min(1.0, sent_score))
            sent_label = "positive" if sent_score > 0.15 else ("negative" if sent_score < -0.15 else "neutral")

            review_date = datetime.utcnow() - timedelta(days=random.randint(1, 365))
            all_reviews.append(Review(
                product_id=product_id,
                dealer_id=dealer_id,
                source='synthetic',
                platform='synthetic',
                rating=float(rating),
                review_body=body,
                reviewer_name=_random_name(),
                skin_type_mentioned=primary_skin,
                verified_purchase=random.random() > 0.5,
                helpful_votes=random.randint(0, 50),
                sentiment_score=round(sent_score, 3),
                sentiment_label=sent_label,
                is_synthetic=True,
                review_date=review_date,
            ))

    # ── Bulk save ────────────────────────────────────────────────────────────
    logger.info("  Saving %d products...", len(all_products))
    db.session.bulk_save_objects(all_products)
    db.session.commit()

    logger.info("  Saving %d sales records...", len(all_sales))
    db.session.bulk_save_objects(all_sales)
    db.session.commit()

    logger.info("  Saving %d inventory records...", len(all_inventory))
    db.session.bulk_save_objects(all_inventory)
    db.session.commit()

    logger.info("  Saving %d reviews...", len(all_reviews))
    db.session.bulk_save_objects(all_reviews)
    db.session.commit()

    logger.info("Synthetic data generation complete for dealer %d", dealer_id)
    return {
        'products_generated': len(all_products),
        'sales_generated': len(all_sales),
        'reviews_generated': len(all_reviews),
    }


def augment_reviews_for_dealer(dealer_id: int):
    """
    Find products for a dealer that have NO reviews and generate some synthetic ones.
    Used for CSV-uploaded products to make the dashboard look populated.
    """
    logger.info("Augmenting reviews for dealer %d...", dealer_id)
    
    # Get products without reviews
    subquery = db.session.query(Review.product_id).filter(Review.dealer_id == dealer_id).subquery()
    products_to_augment = Product.query.filter(
        Product.dealer_id == dealer_id,
        ~Product.product_id.in_(subquery)
    ).all()
    
    if not products_to_augment:
        logger.info("  No products found needing review augmentation.")
        return 0

    all_reviews = []
    
    # Reuse some logic from generate_full_sandbox_data
    for product in products_to_augment:
        product_id = product.product_id
        category = product.category or "Moisturizer"
        price = float(product.price or 500)
        skin = product.skin_suitability or "combination, normal"
        primary_skin = skin.split(',')[0].strip() if skin else 'all'
        
        # Determine quality/rating profile
        if price > 2000: quality = 'high'
        elif price > 800: quality = 'medium'
        else: quality = 'low'
        
        num_reviews = random.randint(5, 12)
        used_bodies = set()
        
        for _ in range(num_reviews):
            r = random.random()
            if quality == 'high':
                rating = 5 if r < 0.65 else (4 if r < 0.90 else 3)
            elif quality == 'medium':
                rating = 5 if r < 0.30 else (4 if r < 0.70 else (3 if r < 0.90 else 2))
            else:
                rating = 4 if r < 0.30 else (3 if r < 0.70 else (2 if r < 0.90 else 1))

            # Pick pool
            if rating == 5:
                pool = REVIEW_TEMPLATES[5].get(primary_skin, REVIEW_TEMPLATES[5]['all']) + REVIEW_TEMPLATES[5]['all']
            elif rating == 4: pool = REVIEW_TEMPLATES[4]
            elif rating == 3: pool = REVIEW_TEMPLATES[3]
            elif rating == 2: pool = REVIEW_TEMPLATES[2]
            else: pool = REVIEW_TEMPLATES[1]
            
            # Category injection
            if rating >= 4:
                cat_pool = REVIEW_TEMPLATES.get("category_specific", {}).get(category, [])
                if cat_pool and random.random() < 0.5:
                    pool = cat_pool
            
            # Select body
            body = random.choice(pool)
            for _ in range(10):
                cand = random.choice(pool)
                if cand not in used_bodies:
                    body = cand
                    break
            used_bodies.add(body)
            
            # Basic sentiment
            sent_score = 0.2 if rating >= 4 else (-0.2 if rating <= 2 else 0.0)
            sent_label = "positive" if sent_score > 0 else ("negative" if sent_score < 0 else "neutral")
            
            all_reviews.append(Review(
                product_id=product_id,
                dealer_id=dealer_id,
                source='synthetic',
                platform='synthetic',
                rating=float(rating),
                review_body=body,
                reviewer_name=_random_name(),
                skin_type_mentioned=primary_skin,
                verified_purchase=True,
                sentiment_score=sent_score,
                sentiment_label=sent_label,
                is_synthetic=True,
                review_date=datetime.utcnow() - timedelta(days=random.randint(1, 180))
            ))

    if all_reviews:
        db.session.bulk_save_objects(all_reviews)
        db.session.commit()
        logger.info("  Generated %d synthetic reviews for %d products.", len(all_reviews), len(products_to_augment))
    
    return len(all_reviews)
