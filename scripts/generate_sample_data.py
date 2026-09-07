"""Generate a synthetic e-commerce dataset with intentional dirty data."""

import os
import random
import csv
from datetime import datetime, timedelta

random.seed(42)

# Configuration
NUM_ROWS = 1000
NUM_DUPLICATES = 30
MISSING_RATE = 0.05  # 5%

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports"]
REGIONS = ["North", "South", "East", "West"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Cash on Delivery"]

# Product names per category
PRODUCTS = {
    "Electronics": ["Laptop", "Phone", "Headphones", "Tablet", "Smartwatch"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress"],
    "Books": ["Novel", "Textbook", "Cookbook", "Biography", "Comic"],
    "Home & Kitchen": ["Blender", "Cookware Set", "Lamp", "Towel Set", "Vacuum"],
    "Sports": ["Yoga Mat", "Dumbbells", "Running Shoes", "Cricket Bat", "Football"],
}

# Price ranges per category (unit_price range)
PRICE_RANGES = {
    "Electronics": (2000, 80000),
    "Clothing": (300, 5000),
    "Books": (100, 2000),
    "Home & Kitchen": (500, 15000),
    "Sports": (200, 10000),
}

# Date range: 12 months
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days


def random_date():
    delta = timedelta(days=random.randint(0, DATE_RANGE_DAYS))
    return (START_DATE + delta).strftime("%Y-%m-%d")


def generate_rows(n):
    rows = []
    customer_pool = [f"CUST_{i:04d}" for i in range(1, 201)]  # 200 customers

    for i in range(1, n + 1):
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        product_id = f"PROD_{CATEGORIES.index(category) * 100 + PRODUCTS[category].index(product) + 1:04d}"

        quantity = random.randint(1, 10)
        price_min, price_max = PRICE_RANGES[category]
        unit_price = round(random.uniform(price_min, price_max), 2)
        discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2)
        revenue = round(quantity * unit_price * (1 - discount), 2)
        cost = round(revenue * random.uniform(0.4, 0.75), 2)

        row = {
            "order_id": f"ORD_{i:05d}",
            "customer_id": random.choice(customer_pool),
            "order_date": random_date(),
            "product_id": product_id,
            "product_category": category,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "cost": cost,
            "payment_method": random.choice(PAYMENT_METHODS),
            "region": random.choice(REGIONS),
        }
        rows.append(row)

    return rows


def inject_missing_values(rows, rate):
    """Randomly blank out values in quantity, unit_price, discount, region."""
    nullable_fields = ["quantity", "unit_price", "discount", "region"]
    count = 0
    for row in rows:
        for field in nullable_fields:
            if random.random() < rate:
                row[field] = ""
                count += 1
    return rows, count


def inject_outliers(rows, n=15):
    """Inject a few extreme revenue values."""
    indices = random.sample(range(len(rows)), min(n, len(rows)))
    for idx in indices:
        rows[idx]["revenue"] = round(random.uniform(500000, 2000000), 2)
    return rows


def inject_duplicates(rows, n):
    """Append n duplicate rows picked randomly from existing rows."""
    dupes = random.choices(rows, k=n)
    # Give duplicates new order_ids so they look like data-entry dupes
    # but keep the same data otherwise
    result = []
    for d in dupes:
        dupe = dict(d)
        result.append(dupe)
    return rows + result


def main():
    os.makedirs("data", exist_ok=True)

    rows = generate_rows(NUM_ROWS - NUM_DUPLICATES)
    rows = inject_outliers(rows)
    rows, missing_count = inject_missing_values(rows, MISSING_RATE)
    rows = inject_duplicates(rows, NUM_DUPLICATES)

    # Shuffle to mix duplicates in
    random.shuffle(rows)

    output_path = os.path.join("data", "sample_ecommerce.csv")
    fieldnames = [
        "order_id", "customer_id", "order_date", "product_id",
        "product_category", "quantity", "unit_price", "discount",
        "revenue", "cost", "payment_method", "region",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {output_path}")
    print(f"  Intentional missing values: ~{missing_count}")
    print(f"  Intentional duplicates: {NUM_DUPLICATES}")
    print(f"  Intentional outliers: 15")


if __name__ == "__main__":
    main()
