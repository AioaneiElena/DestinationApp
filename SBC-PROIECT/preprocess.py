import pandas as pd

df = pd.read_csv("destinations.csv", encoding="latin1")

def parse_tourists(val):
    if isinstance(val, str):
        val = val.lower().strip()
        if "million" in val:
            try:
                return float(val.replace("million", "").strip()) * 1_000_000
            except:
                return None
        elif "-" in val:  
            try:
                parts = val.split("-")
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return ((low + high) / 2) * 1_000_000
            except:
                return None
    return None

df["Tourists"] = df["Approximate Annual Tourists"].apply(parse_tourists)

def tourists_category(val):
    if val is None or pd.isna(val):
        return "Unknown"
    if val < 3_000_000:
        return "Low"
    elif val <= 10_000_000:
        return "Medium"
    else:
        return "High"

df["Tourists_Category"] = df["Tourists"].apply(tourists_category)

df["Cost_Category"] = df["Cost of Living"].str.lower().map({
    "low": "Low",
    "medium": "Medium",
    "medium-high": "High",
    "high": "High"
})

def categorize_safety(val):
    if not isinstance(val, str):
        return "Unknown"
    val = val.lower()
    if any(x in val for x in ["dangerous", "crime", "careful", "at night"]):
        return "Low"
    elif any(x in val for x in ["pickpocket", "watch out", "crowded"]):
        return "Medium"
    elif any(x in val for x in ["very safe", "minimal risks", "safe"]):
        return "High"
    return "Unknown"

df["Safety_Category"] = df["Safety"].apply(categorize_safety)

def extract_foods(food_str):
    if not isinstance(food_str, str):
        return []
    return [f.strip().lower() for f in food_str.split(",")]

df["Famous Foods List"] = df["Famous Foods"].apply(extract_foods)

all_foods = set()
for flist in df["Famous Foods List"]:
    all_foods.update(flist)

for food in all_foods:
    df[f"has_{food.replace(' ', '_')}"] = df["Famous Foods List"].apply(lambda l: food in l)

def classify_language_group(lang):
    lang = str(lang).lower()
    if lang in ["romanian", "italian", "spanish", "portuguese", "french"]:
        return "Romanic"
    elif lang in ["english", "german", "dutch"]:
        return "Germanic"
    elif lang in ["russian", "ukrainian", "polish", "serbian"]:
        return "Slavic"
    else:
        return "Other"

df["Language_Group"] = df["Language"].apply(classify_language_group)

def is_similar_to_romanian(lang):
    lang = str(lang).lower()
    if lang in ["romanian", "italian", "spanish", "portuguese", "french"]:
        return "Yes"
    return "No"

df["Is_Similar_Language"] = df["Language"].apply(is_similar_to_romanian)

def group_category(val):
    val = str(val).lower()
    if any(x in val for x in ["beach", "coastal"]):
        return "Coastal"
    elif any(x in val for x in ["city", "town", "village"]):
        return "Urban"
    elif any(x in val for x in ["park", "forest", "valley", "mountain"]):
        return "Nature"
    elif any(x in val for x in ["church", "mosque", "castle", "fortress", "monastery"]):
        return "Historic"
    elif any(x in val for x in ["museum", "opera", "cultural", "palace"]):
        return "Cultural"
    elif any(x in val for x in ["spa", "geothermal"]):
        return "Wellness"
    else:
        return "Other"

df["Category_Grouped"] = df["Category"].apply(group_category)


keep_cols = [
    "Destination", "Country", "Region", "Category_Grouped",
    "Tourists_Category", "Cost_Category", "Language_Group", "Is_Similar_Language",
    "Currency", "Majority Religion", "Best Time to Visit",
    "Safety_Category"
] + [col for col in df.columns if col.startswith("has_")]


df_clean = df[keep_cols]

df_clean.to_csv("destinations_clean.csv", index=False, encoding="utf-8")
print("✔ Fișierul 'destinations_clean.csv' a fost generat cu succes.")
