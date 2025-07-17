import streamlit as st
import pandas as pd
import json
import math
import random
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

df = pd.read_csv("destinations_clean.csv", encoding="ISO-8859-1")
with open("questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

TARGET = "Destination"
question_columns = list(QUESTIONS.keys())

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def info_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])
    return total_entropy - weighted_entropy

features = [
    "Country", "Region", "Category_Grouped", "Tourists_Category", "Cost_Category",
    "Language_Group", "Is_Similar_Language", "Currency",
    "Majority Religion", "Safety_Category"
]
df_quiz = df[features + [TARGET]].dropna()
info_gains = {f: info_gain(df_quiz, f, TARGET) for f in features}
ordered_features = sorted(info_gains, key=info_gains.get, reverse=True)

# === Utility for recommender ===
def normalize_string(s):
    if isinstance(s, str):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()
    return ""

class DestinationRecommender:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, encoding="utf-8")
        self._prepare_features()

    def _prepare_features(self):
        self.df.dropna(subset=["Destination"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.categorical_features = pd.get_dummies(self.df.drop(columns=["Destination"]))

    def _validate_destinations(self, destination_names):
        df_names = self.df["Destination"].astype(str).apply(normalize_string).tolist()
        valid_destinations = []
        invalid_destinations = []
        for name in destination_names:
            name_normalized = normalize_string(name)
            if name_normalized in df_names:
                matched_name = self.df.loc[
                    self.df["Destination"].astype(str).apply(normalize_string) == name_normalized,
                    "Destination"
                ].values[0]
                valid_destinations.append(matched_name)
            else:
                invalid_destinations.append(name)
        if invalid_destinations:
            raise ValueError(f"Destinations not found: {', '.join(invalid_destinations)}")
        return valid_destinations

    def _get_profile(self, destination_names):
        indices = self.df[self.df["Destination"].isin(destination_names)].index
        return self.categorical_features.loc[indices].mean()

    def _calculate_similarity(self, profile, other_indices):
        return cosine_similarity(
            profile.values.reshape(1, -1),
            self.categorical_features.loc[other_indices]
        ).flatten()

    def _get_common_attributes(self, profile, idx):
        destination_vector = self.categorical_features.loc[idx]
        return [col for col in self.categorical_features.columns if profile[col] >= 0.5 and destination_vector[col] == 1]

    def _get_differences(self, profile, idx):
        destination_vector = self.categorical_features.loc[idx]
        return [col for col in self.categorical_features.columns if profile[col] != destination_vector[col]]

    def get_recommendations(self, destination_names, top_n=5, extended=False):
        valid_destinations = self._validate_destinations(destination_names)
        if len(valid_destinations) < 3:
            raise ValueError("Please provide at least 3 valid destination names.")

        profile = self._get_profile(valid_destinations)
        normalized_valid = [normalize_string(d) for d in valid_destinations]
        other_indices = self.df[
            ~self.df["Destination"].astype(str).apply(normalize_string).isin(normalized_valid)
        ].index

        similarities = self._calculate_similarity(profile, other_indices)
        n_results = top_n if not extended else min(10, len(similarities))
        top_indices = np.argsort(similarities)[-n_results:][::-1]

        recommendations = []
        for i in top_indices:
            idx = other_indices[i]
            destination_name = self.df.loc[idx, "Destination"]
            common_attrs = self._get_common_attributes(profile, idx)
            differences = self._get_differences(profile, idx)
            recommendations.append({
                "destination": destination_name,
                "similarity_score": float(similarities[i]),
                "num_common_attributes": len(common_attrs),
                "num_differences": len(differences),
                "common_attributes": common_attrs,
                "differences": differences
            })

        return recommendations

st.sidebar.title("🧭 Navigation")
pagina = st.sidebar.radio("Choose a page:", ["🧠 Guess the destination", "📝 Quiz about a destination", "🔍 Similar Destinations"])

# === PAGE 1: Guess the destination ===
def pagina_ghiceste():
    st.title("🧠 I will guess the destination you're thinking of!")
    st.markdown("Think of a real tourist destination in Europe.")

    remaining = df.copy()
    for col in question_columns:
        if len(remaining) <= 1:
            break

        q = QUESTIONS[col]
        text = q["text"]
        choices = q["choices"]
        mapping = q.get("mapping", {c: c for c in choices})

        answer = st.selectbox(f"👉 {text}", ["Doesn't matter"] + choices, key=col)
        if answer != "Doesn't matter":
            user_val = mapping[answer]
            remaining = remaining[remaining[col] == user_val]

        if remaining.empty:
            st.error("❌ No destination matches your answers.")
            return

    if len(remaining) == 1:
        row = remaining.iloc[0]
        st.success(f"✅ You were thinking of: **{row['Destination']}** ({row['Country']})")
    elif len(remaining) > 1:
        st.warning(f"🤔 Narrowed down to {len(remaining)} destinations. Extra questions...")
        asked_extras = set()
        for _ in range(3):
            if len(remaining) <= 1:
                break
            candidate_cols = [
                col for col in df.columns
                if col not in question_columns and col not in asked_extras
                and col not in ["Destination", "Country", "Region"]
                and remaining[col].nunique() > 1
            ]
            if not candidate_cols:
                break
            col = candidate_cols[0]
            asked_extras.add(col)
            unique_vals = sorted(remaining[col].dropna().unique().tolist())
            val = st.selectbox(f"🔎 What is the value of '{col.replace('_',' ').capitalize()}'?",
                               ["Doesn't matter"] + unique_vals, key="extra_"+col)
            if val != "Doesn't matter":
                remaining = remaining[remaining[col] == val]

        if len(remaining) == 1:
            row = remaining.iloc[0]
            st.success(f"✅ You were thinking of: **{row['Destination']}** ({row['Country']})")
        else:
            st.error(f"⚠ Possible remaining destinations: {len(remaining)}")
            st.dataframe(remaining[["Destination", "Country", "Category_Grouped"]].head(5))

# === PAGE 2: Quiz ===
def pagina_quiz():
    st.title("📝 Quiz about a destination")
    if "quiz_data" not in st.session_state or st.button("🔁 Try another destination"):
        row = df_quiz.sample(1).iloc[0]
        st.session_state.quiz_data = {
            "row": row,
            "answers": {},
            "submitted": False
        }

    row = st.session_state.quiz_data["row"]
    true_destination = row[TARGET]
    st.info(f"📍 You are being tested on: **{true_destination}**")

    with st.form("quiz_form"):
        user_answers = {}
        for feature in ordered_features:
            correct_value = row[feature]
            options = df_quiz[feature].dropna().unique().tolist()
            options = list(set(options) - {correct_value})
            sampled = random.sample(options, min(2, len(options))) + [correct_value]
            random.shuffle(sampled)
            user_answers[feature] = st.radio(
                f"What is the '{feature.replace('_', ' ').capitalize()}' of {true_destination}?",
                sampled,
                key=feature
            )
        submitted = st.form_submit_button("✅ Check")
        if submitted:
            st.session_state.quiz_data["answers"] = user_answers
            st.session_state.quiz_data["submitted"] = True

    if st.session_state.quiz_data["submitted"]:
        score = 0
        st.subheader("📊 Results:")
        for feature in ordered_features:
            correct = row[feature]
            user_val = st.session_state.quiz_data["answers"][feature]
            if user_val == correct:
                st.success(f"✅ {feature}: Correct ({user_val})")
                score += 1
            else:
                st.error(f"❌ {feature}: You chose **{user_val}**, the correct answer was **{correct}**")
        st.markdown(f"## 🏁 Final score: **{score} / {len(ordered_features)}**")

# === PAGE 3: Recommendations ===
def show_recommend_page():
    st.title("🔍 Similar Destinations")
    destinations_input = st.text_input("Enter destinations you like (comma-separated):", "Rome, Florence, Venice")
    if st.button("Get Recommendations"):
        try:
            destinations = [d.strip() for d in destinations_input.split(",") if d.strip()]
            recommender = DestinationRecommender("destinations_clean.csv")
            recommendations = recommender.get_recommendations(destinations)
            for i, rec in enumerate(recommendations, 1):
                st.subheader(f"Recommendation {i}: {rec['destination']}")
                st.write(f"Similarity Score: {rec['similarity_score']:.2f}")
                st.write("Common Attributes:")
                for attr in rec['common_attributes']:
                    formatted_attr = attr.replace("_", " ").title()
                    st.write(f"- {formatted_attr}")
                st.write("Differences:")
                for diff in rec['differences']:
                    formatted_diff = diff.replace("_", " ").title()
                    st.write(f"- {formatted_diff}")
        except ValueError as ve:
            st.error(str(ve))

# === Page router ===
if pagina == "🧠 Guess the destination":
    pagina_ghiceste()
elif pagina == "📝 Quiz about a destination":
    pagina_quiz()
elif pagina == "🔍 Similar Destinations":
    show_recommend_page()