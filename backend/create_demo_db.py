"""
backend/create_demo_db.py  — 50 Indian cosmetic products, 2 demo dealers.
Run at Render build time: python create_demo_db.py
"""
import json, os, sys, random
from datetime import datetime, timedelta
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND_DIR))
os.environ.setdefault('SECRET_KEY', 'biagiotti-demo-secret-2026')
os.environ.setdefault('FLASK_DEBUG', 'false')

from flask import Flask
from database.models import (db, Dealer, Product, Inventory, Sale, Review,
                              AnalysisResult, HarmfulChemical, Notification, DashboardCache)

DB_PATH = _BACKEND_DIR / 'demo_store.sqlite'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'biagiotti-demo-secret-2026'
db.init_app(app)

# (pid, name, brand, category, price, cost, ingredients, skin_suitability, safety_score)
PRODUCTS = [
    ("P001","Radiance Glow Serum","Biagiotti Glow","Serum",1299,520,"Niacinamide, Hyaluronic Acid, Vitamin C, Glycerin, Aloe Vera","All Skin Types",92.0),
    ("P002","Intense Moisture Cream","Biagiotti Luxe","Moisturizer",899,360,"Shea Butter, Ceramides, Peptides, Squalane, Rose Water","Dry, Normal",95.0),
    ("P003","Matte Velvet Foundation","Biagiotti Color","Makeup",1499,600,"Titanium Dioxide, Zinc Oxide, Dimethicone, Talc, Mica","Oily, Combination",78.0),
    ("P004","Deep Pore Cleanser","Deep Clean","Cleanser",499,180,"Salicylic Acid, Tea Tree Oil, Witch Hazel, Aloe Vera, Panthenol","Oily, Acne-Prone",88.0),
    ("P005","SPF 50+ Invisible Sunscreen","Safe Shield","Suncare",799,320,"Avobenzone, Octinoxate, Zinc Oxide, Aloe Vera, Vitamin E","All Skin Types",83.0),
    ("P006","Retinol Night Repair","Biagiotti Science","Treatment",1899,760,"Retinol 0.3%, Peptides, Ceramides, Niacinamide, Hyaluronic Acid","Mature, Normal",87.0),
    ("P007","Rose Petal Toner","Petal Fresh","Toner",599,220,"Rose Water, Glycerin, Witch Hazel, Hyaluronic Acid, Allantoin","All Skin Types",96.0),
    ("P008","Charcoal Detox Mask","Clear Skin","Mask",699,280,"Activated Charcoal, Kaolin Clay, Tea Tree Oil, Bentonite, Aloe Vera","Oily, Combination",89.0),
    ("P009","Caffeine Eye Rescue Serum","Biagiotti Science","Eye Care",1199,480,"Caffeine 5%, Peptides, Hyaluronic Acid, Vitamin K, Aloe Vera","All Skin Types",91.0),
    ("P010","Keratin Repair Shampoo","Silky Strands","Hair Care",649,260,"Keratin Protein, Argan Oil, Biotin, Panthenol, Glycerin","Damaged Hair",90.0),
    ("P011","Plump & Shine Lip Gloss","Biagiotti Color","Lip Care",449,160,"Castor Oil, Vitamin E, Shea Butter, Beeswax, Peppermint Oil","All Lip Types",94.0),
    ("P012","Brightening Body Lotion","Glow Lab","Body Care",749,300,"Kojic Acid, Vitamin C, Niacinamide, Shea Butter, Squalane","All Skin Types",86.0),
    ("P013","Hydra-Boost Serum","Pure Essence","Serum",1099,440,"Hyaluronic Acid 2%, Ceramides, Panthenol, Aloe Vera, Glycerin","Dry, Dehydrated",97.0),
    ("P014","Silk Touch Moisturizer","Biagiotti Luxe","Moisturizer",999,400,"Collagen Peptides, Jojoba Oil, Squalane, Vitamin E, Rose Hip","Normal, Combination",93.0),
    ("P015","HD Concealer Palette","Biagiotti Color","Makeup",1299,520,"Titanium Dioxide, Talc, Mica, Dimethicone, Cyclopentasiloxane","All Skin Types",80.0),
    ("P016","Gentle Foam Cleanser","Pure Essence","Cleanser",549,200,"Glycerin, Aloe Vera, Chamomile Extract, Panthenol, Allantoin","Sensitive, Dry",98.0),
    ("P017","Tinted Sunscreen SPF 40","Safe Shield","Suncare",899,360,"Zinc Oxide, Iron Oxides, Niacinamide, Glycerin, Dimethicone","All Skin Types",85.0),
    ("P018","Vitamin C Brightening Serum","Glow Lab","Serum",1399,560,"Ascorbic Acid 15%, Ferulic Acid, Vitamin E, Hyaluronic Acid, Niacinamide","Dull, Hyperpigmented",88.0),
    ("P019","AHA BHA Exfoliating Toner","Biagiotti Science","Toner",799,320,"Glycolic Acid 5%, Salicylic Acid 2%, Lactic Acid, Aloe Vera, Panthenol","Oily, Acne-Prone",82.0),
    ("P020","Overnight Glow Mask","Biagiotti Glow","Mask",1199,480,"Bakuchiol, Niacinamide, Hyaluronic Acid, Ceramides, Vitamin C","All Skin Types",94.0),
    ("P021","Peptide Eye Cream","Biagiotti Luxe","Eye Care",1499,600,"Matrixyl 3000, Argireline, Hyaluronic Acid, Vitamin K, Caffeine","Mature, All Types",92.0),
    ("P022","Argan Oil Hair Mask","Silky Strands","Hair Care",799,320,"Argan Oil, Keratin, Biotin, Vitamin E, Panthenol","Dry, Damaged Hair",95.0),
    ("P023","Matte Liquid Lipstick","Biagiotti Color","Lip Care",649,260,"Isododecane, Dimethicone, Vitamin E, Jojoba Oil, Beeswax","All Lip Types",85.0),
    ("P024","Firming Body Butter","Pure Essence","Body Care",899,360,"Shea Butter, Cocoa Butter, Collagen, Vitamin E, Almond Oil","All Skin Types",96.0),
    ("P025","Salicylic Acid Spot Treatment","Clear Skin","Treatment",499,200,"Salicylic Acid 2%, Tea Tree Oil, Niacinamide, Zinc, Aloe Vera","Oily, Acne-Prone",86.0),
    ("P026","Niacinamide 10% Serum","Biagiotti Science","Serum",999,400,"Niacinamide 10%, Zinc 1%, Hyaluronic Acid, Glycerin, Panthenol","Oily, Large Pores",95.0),
    ("P027","Collagen Boost Moisturizer","Biagiotti Luxe","Moisturizer",1199,480,"Marine Collagen, Peptides, Squalane, Ceramides, Vitamin E","Mature, Dry",91.0),
    ("P028","Micellar Cleansing Water","Petal Fresh","Cleanser",449,180,"Micellar Micelles, Glycerin, Rose Water, Allantoin, Panthenol","All Skin Types",98.0),
    ("P029","CC Cream SPF 30","Biagiotti Color","Makeup",1099,440,"Titanium Dioxide, Zinc Oxide, Niacinamide, Hyaluronic Acid, Vitamin C","All Skin Types",84.0),
    ("P030","Biotin Hair Growth Serum","Silky Strands","Hair Care",899,360,"Biotin, Caffeine, Castor Oil, Saw Palmetto, Peptides","Thinning Hair",88.0),
    ("P031","Kumkumadi Face Oil","Pure Essence","Serum",1599,640,"Kumkumadi Oil, Saffron, Turmeric, Sandalwood, Rose Hip Oil","All Skin Types",93.0),
    ("P032","Clay Pore Refining Mask","Clear Skin","Mask",649,260,"Kaolin Clay, Bentonite, Tea Tree Oil, Witch Hazel, Zinc","Oily, Combination",91.0),
    ("P033","Sunflower SPF 60 Sunscreen","Safe Shield","Suncare",999,400,"Sunflower Extract, Zinc Oxide, Tinosorb S, Niacinamide, Vitamin E","All Skin Types",87.0),
    ("P034","Under Eye Dark Circle Cream","Biagiotti Science","Eye Care",1299,520,"Vitamin K, Caffeine, Hyaluronic Acid, Peptides, Arnica Extract","All Skin Types",90.0),
    ("P035","Lip Plumping Serum","Biagiotti Glow","Lip Care",799,320,"Hyaluronic Acid, Collagen, Peptides, Vitamin E, Peppermint","All Lip Types",92.0),
    ("P036","Ubtan Brightening Scrub","Pure Essence","Treatment",699,280,"Turmeric, Chickpea Flour, Sandalwood, Rose Water, Almond Oil","All Skin Types",97.0),
    ("P037","Multani Mitti Face Pack","Glow Lab","Mask",399,160,"Fullers Earth, Neem, Turmeric, Rose Water, Aloe Vera","Oily, Normal",99.0),
    ("P038","Ceramide Barrier Repair Cream","Biagiotti Luxe","Moisturizer",1499,600,"Ceramide NP, Ceramide AP, Cholesterol, Peptides, Hyaluronic Acid","Sensitive, Dry",96.0),
    ("P039","Micellar Rose Toner","Petal Fresh","Toner",549,220,"Rose Water, Hyaluronic Acid, Glycerin, Allantoin, Aloe Vera","All Skin Types",97.0),
    ("P040","Scalp Detox Shampoo","Silky Strands","Hair Care",749,300,"Salicylic Acid, Tea Tree Oil, Zinc Pyrithione, Biotin, Glycerin","Oily Scalp",85.0),
    ("P041","Watermelon Hydrating Serum","Glow Lab","Serum",1099,440,"Watermelon Extract, Hyaluronic Acid, Niacinamide, Aloe Vera, Glycerin","All Skin Types",96.0),
    ("P042","Kojic Acid Brightening Cream","Biagiotti Science","Treatment",1199,480,"Kojic Acid 2%, Vitamin C, Niacinamide, Licorice Extract, Glycerin","Hyperpigmented",79.0),
    ("P043","Liquid Highlighter","Biagiotti Color","Makeup",899,360,"Mica, Isododecane, Dimethicone, Vitamin E, Aloe Vera","All Skin Types",88.0),
    ("P044","Neem Purifying Face Wash","Deep Clean","Cleanser",349,140,"Neem Extract, Tulsi, Tea Tree Oil, Aloe Vera, Glycerin","Oily, Acne-Prone",97.0),
    ("P045","Hyaluronic Tinted Lip Balm","Biagiotti Glow","Lip Care",549,220,"Hyaluronic Acid, Shea Butter, Vitamin E, Beeswax, Jojoba Oil","All Lip Types",97.0),
    ("P046","Body Brightening Serum","Glow Lab","Body Care",1099,440,"Vitamin C, Niacinamide, Kojic Acid, Glycerin, Squalane","All Skin Types",89.0),
    ("P047","Charcoal Detox Toner","Clear Skin","Toner",649,260,"Activated Charcoal, Salicylic Acid, Witch Hazel, Niacinamide, Aloe Vera","Oily, Combination",87.0),
    ("P048","Retinol Eye Serum","Biagiotti Science","Eye Care",1699,680,"Retinol 0.1%, Peptides, Caffeine, Hyaluronic Acid, Vitamin E","Mature Skin",84.0),
    ("P049","Almond & Milk Body Lotion","Pure Essence","Body Care",649,260,"Sweet Almond Oil, Milk Protein, Shea Butter, Vitamin E, Glycerin","All Skin Types",98.0),
    ("P050","Bhringraj Hair Oil","Silky Strands","Hair Care",549,220,"Bhringraj Extract, Coconut Oil, Amla, Brahmi, Castor Oil","All Hair Types",99.0),
    # 30 MORE PRODUCTS (P051-P080)
    # --- 5 UNSAFE products (safety < 70, harmful ingredients) ---
    ("P051","Fair & Glow Whitening Cream","FairSkin","Treatment",599,220,"Hydroquinone 2%, Mercury Compound, Parabens, Fragrance","All Skin Types",42.0),
    ("P052","Bleach Brightening Cream","QuickFair","Treatment",449,160,"Ammonium Persulfate, Formaldehyde, Sodium Lauryl Sulfate","Oily, Normal",38.0),
    ("P053","Chemical Hair Relaxer","StraightPro","Hair Care",899,360,"Formaldehyde 3%, Sodium Hydroxide, Mineral Oil, Parabens","All Hair Types",35.0),
    ("P054","Anti-Ageing Fairness Serum","AgelessFair","Serum",1299,520,"Hydroquinone, Tretinoin, Methylparaben, Propylparaben","Mature Skin",45.0),
    ("P055","Instant Glow Face Pack","GlowNow","Mask",349,140,"Mercury Chloride, Bleaching Agents, Formaldehyde","All Skin Types",32.0),
    # --- 5 UNDERSTOCK (popular, running low) ---
    ("P056","Saffron Glow Face Cream","Pure Essence","Moisturizer",799,320,"Saffron Extract, Vitamin C, Hyaluronic Acid, Aloe Vera","All Skin Types",94.0),
    ("P057","Green Tea Antioxidant Serum","Glow Lab","Serum",1199,480,"Green Tea Extract, EGCG, Vitamin E, Niacinamide, Glycerin","All Skin Types",96.0),
    ("P058","Papaya Enzyme Cleanser","Deep Clean","Cleanser",499,200,"Papaya Extract, AHA, Glycerin, Aloe Vera, Panthenol","Normal, Combination",91.0),
    ("P059","Rose Hip Overnight Cream","Biagiotti Luxe","Moisturizer",1399,560,"Rose Hip Oil, Retinol 0.1%, Peptides, Ceramides, Vitamin E","Dry, Mature",89.0),
    ("P060","Charcoal Peel Off Mask","Clear Skin","Mask",549,220,"Activated Charcoal, PVA, Tea Tree Oil, Witch Hazel, Zinc","Oily, Combination",88.0),
    # --- 5 OVERSTOCK (luxury, slow moving) ---
    ("P061","Luxury Gold Face Serum","Biagiotti Luxe","Serum",2999,1200,"Gold Peptides, Collagen, Hyaluronic Acid, Vitamin C, Squalane","Mature, Dry",97.0),
    ("P062","Diamond Glow Moisturizer","Biagiotti Luxe","Moisturizer",2499,1000,"Diamond Powder, Ceramides, Peptides, Rose Hip, Vitamin E","Mature, Normal",96.0),
    ("P063","Platinum Anti-Ageing Cream","Biagiotti Science","Treatment",3499,1400,"Platinum Peptides, Retinol 0.5%, Vitamin C, Collagen, Ceramides","Mature Skin",90.0),
    ("P064","Pearl Radiance Eye Cream","Biagiotti Luxe","Eye Care",1999,800,"Pearl Extract, Caffeine, Peptides, Hyaluronic Acid, Vitamin K","Mature, All Types",95.0),
    ("P065","Caviar Repair Hair Mask","Silky Strands","Hair Care",1799,720,"Caviar Extract, Keratin, Argan Oil, Biotin, Silk Proteins","Dry, Damaged Hair",98.0),
    # --- 15 NORMAL stock, diverse categories ---
    ("P066","Turmeric Glow Face Wash","Glow Lab","Cleanser",399,160,"Turmeric Extract, Neem, Aloe Vera, Glycerin, Panthenol","All Skin Types",97.0),
    ("P067","Aloe Vera Soothing Gel","Pure Essence","Moisturizer",349,140,"Aloe Vera 99%, Allantoin, Panthenol, Glycerin","Sensitive, All Types",99.0),
    ("P068","SPF 30 Daily Moisturizer","Safe Shield","Suncare",699,280,"Zinc Oxide, Niacinamide, Hyaluronic Acid, Ceramides, Vitamin E","All Skin Types",90.0),
    ("P069","Glycolic Acid Toner","Biagiotti Science","Toner",899,360,"Glycolic Acid 7%, Aloe Vera, Hyaluronic Acid, Panthenol","Dull, Uneven Skin",85.0),
    ("P070","Nourishing Foot Cream","Pure Essence","Body Care",449,180,"Urea 10%, Shea Butter, Tea Tree Oil, Aloe Vera, Vitamin E","All Skin Types",96.0),
    ("P071","Mango Butter Lip Balm","Petal Fresh","Lip Care",299,120,"Mango Butter, Beeswax, Vitamin E, Coconut Oil, Honey Extract","All Lip Types",98.0),
    ("P072","Protein Hair Conditioner","Silky Strands","Hair Care",599,240,"Wheat Protein, Silk Amino Acids, Argan Oil, Panthenol, Glycerin","All Hair Types",94.0),
    ("P073","Anti-Pollution Face Mist","Glow Lab","Toner",799,320,"Antioxidant Complex, Rose Water, Hyaluronic Acid, Vitamin C","Urban Skin",93.0),
    ("P074","Mineral Sunscreen SPF 50","Safe Shield","Suncare",1199,480,"Zinc Oxide 20%, Titanium Dioxide, Aloe Vera, Vitamin E","Sensitive, All Types",96.0),
    ("P075","Bakuchiol Retinol Alternative","Biagiotti Science","Serum",1699,680,"Bakuchiol 1%, Niacinamide, Hyaluronic Acid, Ceramides, Peptides","Sensitive, Mature",98.0),
    ("P076","Mandelic Acid Exfoliator","Biagiotti Science","Treatment",1099,440,"Mandelic Acid 10%, Aloe Vera, Panthenol, Hyaluronic Acid","Sensitive, Acne-Prone",87.0),
    ("P077","Coconut Milk Hair Serum","Silky Strands","Hair Care",699,280,"Coconut Milk, Argan Oil, Vitamin E, Silk Proteins, Glycerin","Frizzy, Dry Hair",97.0),
    ("P078","Licorice Brightening Cream","Glow Lab","Moisturizer",999,400,"Licorice Extract, Niacinamide, Vitamin C, Hyaluronic Acid","Hyperpigmented",95.0),
    ("P079","Azelaic Acid Glow Serum","Biagiotti Science","Serum",1299,520,"Azelaic Acid 10%, Niacinamide 5%, Hyaluronic Acid, Aloe Vera","Acne-Prone, Rosacea",91.0),
    ("P080","Amla & Hibiscus Hair Rinse","Silky Strands","Hair Care",499,200,"Amla Extract, Hibiscus, Bhringraj, Brahmi, Aloe Vera","All Hair Types",99.0),
]

REVIEWS_POS = [
    "Absolutely love this! My skin transformed in 2 weeks.",
    "Best product I've stocked this year. Customers keep reordering.",
    "Premium quality at a great price. Flying off the shelves!",
    "Works beautifully on Indian skin tones. Highly recommended.",
    "My customers with dry skin swear by this. 5 stars easily.",
]
REVIEWS_NEU = [
    "Decent product, does what it says. Average results.",
    "Some customers like it, some don't. Mixed reviews.",
    "Takes time to show results. Patience needed.",
]
REVIEWS_NEG = [
    "A bit heavy for summer. Not ideal for very oily skin.",
    "A few customers reported mild irritation on sensitive skin.",
    "Packaging could be better. Product itself is okay.",
]
NAMES = ["Priya Sharma","Anjali Singh","Deepika Nair","Sunita Patel","Kavitha Reddy",
         "Meena Gupta","Ritu Agarwal","Pooja Menon","Ananya Bose","Shalini Iyer",
         "Divya Pillai","Rekha Choudhary","Nandini Joshi","Lalitha Kumar","Sneha Das"]
CITIES  = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad"]
REGIONS = ["Maharashtra","Delhi NCR","Karnataka","Telangana","Tamil Nadu","Gujarat","West Bengal"]
EVENTS  = ["Regular Sale","Festival Offer","Clearance","New Launch","Reorder","Bulk Order"]
MONTHS  = [(2025,12),(2026,1),(2026,2),(2026,3),(2026,4),(2026,5)]

HARMFUL_CHEMS = [
    ("Parabens (Methylparaben)","4418-26-2","HIGH","Endocrine disruptor","Restricted in EU","Preservative","Paraben",8,"EWG"),
    ("Sodium Lauryl Sulfate","151-21-3","MEDIUM","Skin irritant","Permitted with limits","Surfactant","Sulfate",5,"FDA"),
    ("Formaldehyde","50-00-0","HIGH","Known carcinogen","Banned in EU","Preservative","Aldehyde",9,"IARC"),
    ("Hydroquinone","123-31-9","HIGH","Potential carcinogen","Banned in EU","Skin Lightener","Phenol",8,"WHO"),
    ("Phthalates (DBP)","84-74-2","HIGH","Endocrine disruption","Banned in EU","Plasticizer","Phthalate",8,"EWG"),
]


def make_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    with app.app_context():
        db.create_all()
        print("[INFO] Tables created.")

        # Harmful chemicals
        for row in HARMFUL_CHEMS:
            db.session.add(HarmfulChemical(
                chemical_name=row[0], cas_number=row[1], risk_level=row[2],
                health_risk=row[3], legal_status=row[4], primary_category=row[5],
                sub_category=row[6], severity_score=row[7], source_dataset=row[8]))

        # Dealers
        d1 = Dealer(name="Demo Dealer", email="demo@cosmetic.ai",
                    shop_name="Biagiotti Demo Store", city="Mumbai", phone="9876543210")
        d1.set_password("demo1234")
        d2 = Dealer(name="Test Biagiotti", email="test@biagiotti.com",
                    shop_name="Biagiotti Test Shop", city="Bangalore", phone="9123456780")
        d2.set_password("test1234")
        db.session.add_all([d1, d2])
        db.session.flush()

        for dealer_id in [d1.dealer_id, d2.dealer_id]:
            rng = random.Random(dealer_id * 7)
            top_list, alert_list, cat_counts = [], [], {}
            n_under, n_over, n_harmful, n_crit = 0, 0, 0, 0

            for i, (pid, pname, brand, cat, price, cost, ingr, skin, safety) in enumerate(PRODUCTS):
                db.session.add(Product(
                    product_id=pid, dealer_id=dealer_id,
                    product_name=pname, brand=brand, category=cat,
                    price=float(price), cost_price=float(cost),
                    ingredients=ingr, skin_suitability=skin,
                    country="India", label="Retail", is_verified=True))
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

                # Stock control using pid for explicit demo scenarios
                if pid in ("P056","P057","P058","P059","P060","P051","P052","P053","P054","P055"):
                    stock = rng.randint(5, 25)    # understock + unsafe
                elif pid in ("P061","P062","P063","P064","P065"):
                    stock = rng.randint(180, 350) # luxury overstock
                elif i < 10:
                    stock = rng.randint(5, 28)    # first 10 understock
                elif i < 15:
                    stock = rng.randint(160, 300) # next 5 overstock
                elif i % 7 == 0:
                    stock = rng.randint(8, 35)
                elif i % 9 == 0:
                    stock = rng.randint(155, 250)
                else:
                    stock = rng.randint(55, 140)

                db.session.add(Inventory(
                    product_id=pid, dealer_id=dealer_id,
                    current_stock=float(stock), reorder_level=50.0,
                    lead_time_days=14.0,
                    last_restocked=datetime.utcnow() - timedelta(days=rng.randint(10, 45))))

                if stock < 50: n_under += 1; n_crit += 1
                if stock > 150: n_over += 1

                base = rng.randint(25, 110)
                monthly = []
                for yr, mo in MONTHS:
                    units = max(5, int(base * (1 + MONTHS.index((yr,mo))*0.04) * rng.uniform(0.8,1.2)))
                    db.session.add(Sale(
                        product_id=pid, dealer_id=dealer_id, brand=brand,
                        region=rng.choice(REGIONS), city=rng.choice(CITIES),
                        event_type=rng.choice(EVENTS), year=yr, month=mo,
                        units_sold=float(units), revenue=round(units*price*rng.uniform(0.9,1.0),2),
                        sell_through_pct=round(rng.uniform(0.55,0.95),2),
                        avg_daily_footfall=float(rng.randint(40,200)),
                        sale_date=datetime(yr, mo, rng.randint(1,28))))
                    monthly.append(units)

                ratings, sentiments = [], []
                for _ in range(rng.randint(3, 6)):
                    r = rng.choices([5,4,3,2], weights=[35,40,15,10])[0]
                    if r >= 4: txt=rng.choice(REVIEWS_POS); sl="positive"; ss=rng.uniform(0.55,0.95)
                    elif r==3: txt=rng.choice(REVIEWS_NEU); sl="neutral";  ss=rng.uniform(-0.1,0.2)
                    else:      txt=rng.choice(REVIEWS_NEG); sl="negative"; ss=rng.uniform(-0.6,-0.15)
                    db.session.add(Review(
                        product_id=pid, dealer_id=dealer_id, source="demo",
                        platform=rng.choice(["Nykaa","Amazon","Flipkart","In-Store"]),
                        rating=float(r), review_title=txt[:60], review_body=txt,
                        reviewer_name=rng.choice(NAMES),
                        skin_type_mentioned=rng.choice(["Oily","Dry","Combination","Normal",""]),
                        verified_purchase=rng.random()>0.3,
                        helpful_votes=rng.randint(0,45),
                        sentiment_score=round(ss,3), sentiment_label=sl,
                        is_synthetic=True,
                        review_date=datetime.utcnow()-timedelta(days=rng.randint(5,180))))
                    ratings.append(r); sentiments.append(ss)

                avg_r = round(sum(ratings)/len(ratings),2)
                avg_s = round(sum(sentiments)/len(sentiments),3)
                fore  = [int(monthly[-1]*(1+0.05*j)) for j in range(1,7)]
                days_out = round(stock/max(1,monthly[-1]/30),1)

                # FIXED: correct lowercase values matching DB query filters
                if safety < 70:
                    risk_lvl = 'high'
                    harmful_json = json.dumps([{"name": ingr.split(',')[0].strip(), "severity": "high"}])
                    n_harmful += 1
                elif safety < 85:
                    risk_lvl = 'medium'
                    harmful_json = json.dumps([])
                else:
                    risk_lvl = 'none'
                    harmful_json = json.dumps([])

                stock_st   = 'understock' if stock < 50 else ('overstock' if stock > 150 else 'normal')
                trend_st   = 'increasing' if fore[-1] > monthly[-1]*1.02 else 'stable'
                sout_risk  = 'HIGH' if days_out < 14 else ('MEDIUM' if days_out < 30 else 'LOW')
                ss_status  = 'Safe' if safety >= 85 else ('Moderate' if safety >= 70 else 'Unsafe')

                if stock < 50:
                    alert_list.append({"type":"low_stock","product":pname,
                        "message":f"Only {stock} units. Reorder now.","severity":"high"})
                if safety < 70:
                    alert_list.append({"type":"safety","product":pname,
                        "message":"Contains harmful ingredients. Review before reorder.",
                        "severity":"critical"})

                db.session.add(AnalysisResult(
                    product_id=pid, dealer_id=dealer_id,
                    demand_forecast_json=json.dumps({"forecast":fore,"months":6}),
                    forecast_trend=trend_st,
                    stock_status=stock_st,
                    days_until_stockout=days_out, stockout_risk=sout_risk,
                    skin_type_detected=skin, skin_confidence=round(rng.uniform(0.75,0.98),2),
                    harmful_ingredients_json=harmful_json,
                    safety_score=safety, safety_status=ss_status, risk_level=risk_lvl,
                    stock_decision='RESTOCK' if stock<50 else ('CLEAR' if stock>150 else 'HOLD'),
                    decision_reason='Below reorder level' if stock<50 else ('Excess stock - consider discount' if stock>150 else 'Stock adequate'),
                    priority_score=round(rng.uniform(0.5,0.99),2),
                    recommendations_json=json.dumps([{"action":"Monitor demand","priority":"MEDIUM"}]),
                    avg_rating=avg_r, review_count=len(ratings),
                    sentiment_avg=avg_s, verification_status='verified',
                    analyzed_at=datetime.utcnow()))

                top_list.append({"product_id":pid,"product_name":pname,"brand":brand,
                    "category":cat,"avg_rating":avg_r,"safety_score":safety,"stock":stock})

            # Notifications
            db.session.add(Notification(dealer_id=dealer_id, product_id="P001",
                notif_type="low_stock", severity="high", is_read=False,
                title="⚠️ Low Stock: Radiance Glow Serum",
                message="Only 12 units left. Reorder level is 50."))
            db.session.add(Notification(dealer_id=dealer_id, product_id="P006",
                notif_type="safety_alert", severity="medium", is_read=False,
                title="🔬 Safety: Retinol Night Repair",
                message="Retinol concentration above 0.3% requires dermatologist review for some skin types."))
            db.session.add(Notification(dealer_id=dealer_id, product_id="P010",
                notif_type="restock_reminder", severity="low", is_read=True,
                title="📦 Festival Restock Reminder",
                message="Keratin Repair Shampoo sells 3x during Diwali. Stock up now."))

            top_list.sort(key=lambda x: x["avg_rating"], reverse=True)
            cat_health = [{"category":c,"count":n,"health":"Good" if n>=2 else "Fair"}
                          for c,n in cat_counts.items()]

            db.session.add(DashboardCache(
                dealer_id=dealer_id,
                total_products=len(PRODUCTS),
                understock_count=n_under, overstock_count=n_over,
                harmful_count=n_harmful, critical_alerts=n_crit,
                top_products_json=json.dumps(top_list[:10]),
                alert_feed_json=json.dumps(alert_list[:10]),
                category_health_json=json.dumps(cat_health),
                pipeline_status="done", pipeline_progress=100,
                pipeline_started_at=datetime.utcnow()-timedelta(minutes=5),
                last_updated=datetime.utcnow(),
                next_scheduled=datetime.utcnow()+timedelta(hours=24)))

            print(f"[INFO] Dealer {dealer_id}: {len(PRODUCTS)} products, {n_under} low-stock.")

        db.session.commit()
        print(f"\n✅ demo_store.sqlite ready — {len(PRODUCTS)} products")
        print("   demo@cosmetic.ai / demo1234")
        print("   test@biagiotti.com / test1234")


if __name__ == '__main__':
    make_db()
