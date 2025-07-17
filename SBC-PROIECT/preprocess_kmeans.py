import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_data(df):
    df = df.copy()
  
    transformations = {}
    
    multi_cols = ['Famous Foods', 'Language', 'Best Time to Visit']
    for col in multi_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').str.split(',').apply(
                lambda items: [item.strip() for item in items if item.strip()]
            )
            mlb = MultiLabelBinarizer()
            dummy = pd.DataFrame(
                mlb.fit_transform(df[col]),
                columns=[f"{col}_{cls}" for cls in mlb.classes_],
                index=df.index
            )
            transformations.update(dummy)
            df.drop(columns=[col], inplace=True)
   
    num_cols = ['Safety', 'Cost of Living']
    for col in num_cols:
        if col in df.columns:
            safety_map = {'High': 3, 'Medium': 2, 'Low': 1}
            cost_map = {'High': 3, 'Medium': 2, 'Low': 1}
            
            if col == 'Safety':
                df[col] = df[col].map(safety_map)
            elif col == 'Cost of Living':
                df[col] = df[col].map(cost_map)
           
            km = KMeans(n_clusters=3, random_state=42)
            data_for_clustering = df[[col]].fillna(df[col].mean()).values
            if len(data_for_clustering) > 0:
                labels = km.fit_predict(data_for_clustering)
                df[f"{col}_cluster"] = labels
    
    if transformations:
        df = pd.concat([df, pd.DataFrame(transformations, index=df.index)], axis=1)
    
    keep = [
        'Destination', 'Region', 'Country', 'Category', 'Currency',
        'Majority Religion', 'Safety', 'Cost of Living', 'Climate'
    ]
    keep += [c for c in df.columns if any(c.startswith(prefix) for prefix in multi_cols)]
    keep += [c for c in df.columns if c.endswith('_cluster')]
    
    keep = [col for col in keep if col in df.columns]
    df_preproc = df[keep].copy()
    
    assert df_preproc.shape[0] == df.shape[0], (
        f"Așteptam {df.shape[0]} rânduri, am {df_preproc.shape[0]}"
    )
    
    return df_preproc
