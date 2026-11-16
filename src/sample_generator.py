"""
ShopGraph: Sample Retail Data Generator
Generates realistic e-commerce transaction data mimicking the UCI Online Retail II dataset.
Used as an automated fallback if the raw Excel file is unavailable.
"""

import os
import random
import datetime
import pandas as pd
import numpy as np

def generate_sample_retail_data(
    output_path: str,
    n_records: int = 60000,
    n_customers: int = 1500,
    n_products: int = 400,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating realistic sample retail dataset with {n_records} records...")

    # Realistic product categories and descriptions
    product_templates = [
        ("WHITE HANGING HEART T-LIGHT HOLDER", 2.95, 12),
        ("REGENCY CAKESTAND 3 TIER", 12.75, 4),
        ("JUMBO BAG RED RETROSPOT", 1.95, 20),
        ("ASSORTED COLOUR BIRD ORNAMENT", 1.69, 16),
        ("PARTY BUNTING", 4.95, 8),
        ("LUNCH BAG RED RETROSPOT", 1.65, 15),
        ("SET OF 3 CAKE TINS PANTRY DESIGN", 4.95, 6),
        ("HEART OF WICKER SMALL", 1.65, 10),
        ("NATURAL SLATE HEART CHALKBOARD", 2.95, 12),
        ("HEART OF WICKER LARGE", 2.95, 8),
        ("JUMBO BAG PINK POLKADOT", 1.95, 18),
        ("JAM MAKING SET WITH JARS", 4.25, 6),
        ("RECIPE BOX PANTRY DESIGN", 7.95, 4),
        ("WOODEN PICTURE FRAME WHITE FINISH", 2.55, 10),
        ("VICTORIAN GLASS HANGING T-LIGHT", 1.25, 24),
        ("SET OF 4 PANTRY JELLY MOULDS", 1.25, 12),
        ("RETROSPOT TEA SET CERAMIC 11 PC", 6.95, 4),
        ("PAPER CHAIN KIT 50'S CHRISTMAS", 2.55, 12),
        ("HOT WATER BOTTLE KEEP CALM", 4.95, 8),
        ("CHOCOLATE HOT WATER BOTTLE", 4.95, 8),
        ("SCOTTIE DOG HOT WATER BOTTLE", 4.95, 8),
        ("HAND WARMER BIRD NEST", 2.10, 15),
        ("HAND WARMER OWL DESIGN", 2.10, 15),
        ("DOORMAT RED RETROSPOT", 7.95, 4),
        ("DOORMAT WELCOME TO COUNTRY HOME", 7.95, 4),
        ("BAKING SET 9 PIECE RETROSPOT", 4.95, 6),
        ("GARDENERS KNEELING PAD CUP OF TEA", 1.65, 12),
        ("SET/20 RED RETROSPOT PAPER NAPKINS", 0.85, 30),
        ("BOX OF 6 ASSORTED COLOUR TEASPOONS", 4.25, 6),
        ("PACK OF 72 RETROSPOT CAKE CASES", 0.55, 48),
        ("SMALL POPCORN HOLDER", 0.85, 36),
        ("LARGE POPCORN HOLDER", 1.65, 24),
        ("VINTAGE SNAP CARDS", 0.85, 24),
        ("TRADITIONAL WOODEN CATCH CUP GAME", 1.25, 20),
        ("WOODEN BOX OF DOMINOES", 1.25, 20),
        ("MINI PAINT SET VINTAGE", 0.65, 36),
        ("SET OF 3 HEART STEAMER TRAYS", 2.55, 10),
        ("ZINC FINISH SWEETHEART LETTER RACK", 3.75, 6),
        ("ZINC FOLKART HEART SHELF", 2.95, 8),
        ("ZINC T-LIGHT HOLDER STAR LARGE", 0.95, 24),
    ]

    # Expand to n_products
    catalog = []
    base_stock = 20000
    for i in range(n_products):
        tmpl = product_templates[i % len(product_templates)]
        stock_code = str(base_stock + i)
        desc = tmpl[0] if i < len(product_templates) else f"{tmpl[0]} VARIATION {i // len(product_templates)}"
        base_price = max(0.45, round(tmpl[1] * (0.8 + 0.4 * random.random()), 2))
        std_pack = tmpl[2]
        catalog.append((stock_code, desc, base_price, std_pack))

    # Skewed product popularity (Zipf-like distribution)
    product_weights = np.random.zipf(a=1.4, size=n_products).astype(float)
    product_weights /= product_weights.sum()

    # Customer weights (some wholesale buyers, many retail)
    customer_ids = [12000 + i for i in range(n_customers)]
    customer_types = np.random.choice(["retail", "loyal", "wholesale"], size=n_customers, p=[0.75, 0.20, 0.05])
    customer_weights = np.random.zipf(a=1.3, size=n_customers).astype(float)
    customer_weights /= customer_weights.sum()

    countries = ["United Kingdom"] * 85 + ["Germany", "France", "EIRE", "Spain", "Netherlands", "Switzerland", "Belgium", "Portugal", "Australia"]
    country_weights = [0.88] + [0.12 / (len(countries) - 1)] * (len(countries) - 1)

    customer_country_map = {
        cid: np.random.choice(countries, p=country_weights) for cid in customer_ids
    }

    start_date = datetime.datetime(2009, 12, 1, 8, 30, 0)
    end_date = datetime.datetime(2011, 12, 9, 18, 0, 0)
    time_span_seconds = int((end_date - start_date).total_seconds())

    rows = []
    invoice_num = 490000

    # Build invoices
    while len(rows) < n_records:
        invoice_num += 1
        cid = np.random.choice(customer_ids, p=customer_weights)
        ctype = customer_types[cid - 12000]
        country = customer_country_map[cid]

        # Order timestamp
        inv_time = start_date + datetime.timedelta(seconds=random.randint(0, time_span_seconds))
        inv_str = str(invoice_num)

        # Cancellation chance ~2%
        is_cancellation = (random.random() < 0.02)
        if is_cancellation:
            inv_str = f"C{invoice_num}"

        # Basket size
        if ctype == "wholesale":
            basket_size = random.randint(15, 60)
        elif ctype == "loyal":
            basket_size = random.randint(5, 25)
        else:
            basket_size = random.randint(1, 12)

        chosen_indices = np.random.choice(n_products, size=basket_size, replace=False, p=product_weights)

        for p_idx in chosen_indices:
            p_code, p_desc, p_price, p_pack = catalog[p_idx]

            if ctype == "wholesale":
                qty = p_pack * random.choice([2, 5, 10, 20, 50])
            else:
                qty = random.choice([1, 2, 3, 4, 6, 12, p_pack])

            if is_cancellation:
                qty = -qty

            # Add occasional noise:
            # Null customer ID (~18% chance)
            assigned_cid = None if (random.random() < 0.18) else cid

            # Occasional non-product service charges
            if random.random() < 0.005:
                svc = random.choice([("POST", "POSTAGE", 15.0), ("M", "Manual", 10.0), ("BANK CHARGES", "Bank Charges", 15.0)])
                rows.append({
                    "Invoice": inv_str,
                    "StockCode": svc[0],
                    "Description": svc[1],
                    "Quantity": 1,
                    "InvoiceDate": inv_time,
                    "Price": svc[2],
                    "Customer ID": assigned_cid,
                    "Country": country
                })
            else:
                rows.append({
                    "Invoice": inv_str,
                    "StockCode": p_code,
                    "Description": p_desc,
                    "Quantity": qty,
                    "InvoiceDate": inv_time,
                    "Price": p_price,
                    "Customer ID": assigned_cid,
                    "Country": country
                })

            if len(rows) >= n_records:
                break

    # Add ~1% exact duplicates
    n_dupes = int(len(rows) * 0.01)
    for _ in range(n_dupes):
        rows.append(random.choice(rows).copy())

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Sample dataset saved to: {output_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "data", "sample_retail_data.parquet")
    generate_sample_retail_data(out)
