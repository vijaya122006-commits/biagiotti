"""
fix_reviews_diversity.py
Replaces all repetitive synthetic review bodies + titles in the DB
with a large, diverse pool of realistic Indian-market cosmetic reviews.
Run once from the backend/ directory.
"""
import random
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database.db import db
from database.models import Review

# ─── Expanded review corpus ───────────────────────────────────────────────────

RATING5_OILY = [
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
]

RATING5_DRY = [
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
]

RATING5_SENSITIVE = [
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
]

RATING5_ALL = [
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
]

RATING4_TEMPLATES = [
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
]

RATING3_TEMPLATES = [
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
]

RATING2_TEMPLATES = [
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
]

RATING1_TEMPLATES = [
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
]

REVIEW_TITLES_5 = [
    "Holy grail product!", "Absolutely love this!", "Best skincare purchase ever",
    "Transformed my skin completely", "Exceeded all expectations", "Incredible results!",
    "Must-have for every routine", "So glad I tried this", "Life-changing product",
    "My skin has never looked better", "Highly recommend to everyone", "A true game changer",
    "Worth every rupee", "Obsessed with this product", "Repurchasing forever",
    "5 stars isn't enough", "My skin glows like never before", "Amazing results in 2 weeks",
    "Best decision I ever made for my skin", "Nothing else compares",
]

REVIEW_TITLES_4 = [
    "Really good but minor room for improvement", "Great product overall", "Happy with this purchase",
    "Solid performer", "Good results, minor quibbles", "Works well for my skin type",
    "Nice product, almost perfect", "Would recommend with some notes",
    "Good purchase, not flawless", "Almost five stars", "Pleasantly impressed",
    "A reliable addition to my routine", "Does the job well", "Very good, wish packaging was better",
    "Effective and gentle", "Good value for money", "Would buy again probably",
    "Consistent performance", "Good daily staple", "Liked it but have small complaints",
]

REVIEW_TITLES_3 = [
    "It's okay", "Average product", "Mixed feelings", "Not bad, not great",
    "Middle of the road", "Decent for the price", "Underwhelming but usable",
    "Has its pros and cons", "Take it or leave it", "Nothing special",
    "Passable product", "Just okay for me", "Could be better", "Neutral experience",
    "Not impressed, not disappointed", "Room for major improvement", "Does the basics",
    "Average at best", "Meh", "Three stars feels right",
]

REVIEW_TITLES_2 = [
    "Disappointed", "Didn't work for me", "Below expectations", "Would not recommend",
    "Overhyped product", "Save your money", "Not worth the price", "Frustrating results",
    "Didn't live up to the claims", "Sad purchase", "Not what I expected",
    "Falling short", "Two stars is generous", "Wasted time and money",
    "Poor performer", "Expected much more", "Skip this one",
]

REVIEW_TITLES_1 = [
    "Terrible product!", "Caused a reaction!", "Avoid at all costs", "Worst purchase ever",
    "Returned immediately", "Damaged my skin", "Complete waste of money",
    "Dangerous — caused irritation", "Don't waste your rupees", "Horrific experience",
    "Zero stars if possible", "Broke me out badly", "Misleading claims",
    "Quality control issue", "Allergic reaction — stay away", "Never again",
    "Disaster for my skin", "Money down the drain",
]

SKIN_SPECIFIC_5 = {
    'oily': RATING5_OILY,
    'dry': RATING5_DRY,
    'sensitive': RATING5_SENSITIVE,
    'combination': RATING5_ALL,
    'acne': RATING5_OILY,
    'normal': RATING5_ALL,
    'all': RATING5_ALL,
}


def _get_body(rating: float, skin_type: str) -> str:
    if rating >= 5:
        skin_key = (skin_type or 'all').split(',')[0].strip().lower()
        pool = SKIN_SPECIFIC_5.get(skin_key, RATING5_ALL)
        return random.choice(pool + RATING5_ALL)
    elif rating >= 4:
        return random.choice(RATING4_TEMPLATES)
    elif rating >= 3:
        return random.choice(RATING3_TEMPLATES)
    elif rating >= 2:
        return random.choice(RATING2_TEMPLATES)
    else:
        return random.choice(RATING1_TEMPLATES)


def _get_title(rating: float) -> str:
    if rating >= 5:
        return random.choice(REVIEW_TITLES_5)
    elif rating >= 4:
        return random.choice(REVIEW_TITLES_4)
    elif rating >= 3:
        return random.choice(REVIEW_TITLES_3)
    elif rating >= 2:
        return random.choice(REVIEW_TITLES_2)
    else:
        return random.choice(REVIEW_TITLES_1)


def fix_reviews(app):
    with app.app_context():
        print("Fetching synthetic reviews...")
        reviews = Review.query.filter_by(is_synthetic=True).all()
        print(f"Found {len(reviews)} synthetic reviews. Updating...")

        random.seed(None)  # use system entropy

        # Track per-product used bodies to avoid exact repeats within a product
        used: dict[str, set] = {}

        for i, rev in enumerate(reviews):
            pid = rev.product_id
            if pid not in used:
                used[pid] = set()

            skin = rev.skin_type_mentioned or 'all'

            # Use a unique per-review seed so each review gets a different random draw
            random.seed(f"{pid}-{rev.review_id}-{i}")
            for _ in range(30):
                body = _get_body(rev.rating, skin)
                if body not in used[pid]:
                    break
            used[pid].add(body)

            # Independent seed for title
            random.seed(f"title-{rev.review_id}-{i}")
            rev.review_body = body
            rev.review_title = _get_title(rev.rating)

        # Restore random state to non-deterministic after we're done
        random.seed(None)


        print("Committing changes...")
        db.session.commit()

        # Verify
        from sqlalchemy import func, text
        total = db.session.execute(
            text("SELECT COUNT(*) FROM reviews WHERE is_synthetic=1")
        ).scalar()
        unique_bodies = db.session.execute(
            text("SELECT COUNT(DISTINCT review_body) FROM reviews WHERE is_synthetic=1")
        ).scalar()
        print(f"\n✅ Done! {total} synthetic reviews now have {unique_bodies} unique bodies.")
        print(f"   Diversity ratio: {unique_bodies}/{total} = {unique_bodies/total*100:.1f}% unique")


if __name__ == "__main__":
    from app import app
    fix_reviews(app)
