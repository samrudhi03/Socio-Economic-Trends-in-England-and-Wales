import pandas as pd
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer


df = pd.read_csv("TS053-2021-4.csv")

# reconstruting the table
pivoted_df = df.pivot(
    index="Lower tier local authorities",
    columns="Occupancy rating for rooms (6 categories)",
    values="Observation"
).reset_index()


pivoted_df.to_csv("person per room 2021.csv", index=False)

df = pd.read_csv("TS054-2021-4 (1).csv")

# reconstruting the table
pivoted_df = df.pivot(
    index=["Lower tier local authorities Code", "Lower tier local authorities"],
    columns="Tenure of household (9 categories)",
    values="Observation"
).reset_index()

pivoted_df.to_csv("tenure 2021.csv", index=False)

# loading all the files
tenure_2011 = pd.read_csv("Tenure 2011.csv")
occupancy_2011 = pd.read_csv("person per room 2011.csv")
heating_2011 = pd.read_csv("heating2011.csv")
deprivation_2011 = pd.read_csv("Deprivation2011.csv")
occupancy_2021 = pd.read_csv("person per room 2021.csv")
tenure_2021 = pd.read_csv("tenure 2021.csv")


tenure_2011.head()

tenure_2021.head()

# Renaming columns in tenure_2011 to match tenure_2021
tenure_2011_renamed = tenure_2011.rename(columns={
    'Tenure: Owned: Owned outright; measures: Value': 'Owned: Owns outright',
    'Tenure: Owned: Owned with a mortgage or loan; measures: Value': 'Owned: Owns with a mortgage or loan',
    'Tenure: Private rented: Other; measures: Value': 'Private rented: Other private rented',
    'Tenure: Private rented: Private landlord or letting agency; measures: Value': 'Private rented: Private landlord or letting agency',
    'Tenure: Shared ownership (part owned and part rented); measures: Value': 'Shared ownership: Shared ownership',
    'Tenure: Social rented: Other; measures: Value': 'Social rented: Other social rented',
    'Tenure: Social rented: Rented from council (Local Authority); measures: Value': 'Social rented: Rents from council or Local Authority',
    'Tenure: Living rent free; measures: Value': 'Lives rent free'
})

# Keeping only the columns that exist in tenure_2021
columns_to_keep = [
    'geography',
    'geography code',
    'Lives rent free',
    'Owned: Owns outright',
    'Owned: Owns with a mortgage or loan',
    'Private rented: Other private rented',
    'Private rented: Private landlord or letting agency',
    'Shared ownership: Shared ownership',
    'Social rented: Other social rented',
    'Social rented: Rents from council or Local Authority'
]

tenure_2011_matched = tenure_2011_renamed[columns_to_keep]


tenure_2011_matched.head()


# Renaming first two columns in tenure_2021
tenure_2021 = tenure_2021.rename(columns={
    'Lower tier local authorities Code': 'geography code',
    'Lower tier local authorities': 'geography'
})

tenure_2021.head()

tenure_2011_matched.to_csv('tenure_2011_cleaned.csv', index=False)
tenure_2021.to_csv('tenure_2021_cleaned.csv', index=False)


occupancy_2011.head()

occupancy_2021.head()

# Renaming columns in occupancy_2011 to match occupancy_2021
occupancy_2011_renamed = occupancy_2011.rename(columns={
    'Occupancy Rating: Occupancy rating (rooms) of +1; measures: Value': 'Occupancy rating of rooms: +1',
    'Occupancy Rating: Occupancy rating (rooms) of +2 or more; measures: Value': 'Occupancy rating of rooms: +2 or more',
    'Occupancy Rating: Occupancy rating (rooms) of 0; measures: Value': 'Occupancy rating of rooms: 0',
    'Occupancy Rating: Occupancy rating (rooms) of -1; measures: Value': 'Occupancy rating of rooms: -1',
    'Occupancy Rating: Occupancy rating (rooms) of -2 or less; measures: Value': 'Occupancy rating of rooms: -2 or less'
})

# Keeping only matching columns in occupancy_2011
columns_to_keep = [
    'geography',
    'geography code',
    'Occupancy rating of rooms: +1',
    'Occupancy rating of rooms: +2 or more',
    'Occupancy rating of rooms: -1',
    'Occupancy rating of rooms: -2 or less',
    'Occupancy rating of rooms: 0'
]

occupancy_2011_matched = occupancy_2011_renamed[columns_to_keep]

# Droping "Does not apply" column from occupancy_2021
occupancy_2021_cleaned = occupancy_2021.drop(columns=['Does not apply'])



# Mapping the 'geography code' from occupancy_2011_matched based on the (geography) name
occupancy_2021_cleaned['geography code'] = occupancy_2021_cleaned['Lower tier local authorities'].map(
    occupancy_2011_matched.set_index('geography')['geography code']
)
# Droping rows where 'geography code' is missing (NaN)
occupancy_2021_cleaned = occupancy_2021_cleaned.dropna(subset=['geography code'])

# Renaming the column 'Lower tier local authorities' to 'geography'
occupancy_2021_cleaned = occupancy_2021_cleaned.rename(columns={'Lower tier local authorities': 'geography'})



occupancy_2011_matched.head()


occupancy_2021_cleaned.tail()


occupancy_2011_matched.to_csv('occupancy_2011_cleaned.csv', index=False)
occupancy_2021_cleaned.to_csv('occupancy_2021_cleaned.csv', index=False)


heating_2011.head()

deprivation_2011 = pd.read_csv('Deprivation2011.csv')

# Renaming the columns
deprivation_2011.columns = [
    "date",
    "geography",
    "geography code",
    "rural urban",
    "total households",
    "not deprived",
    "deprived 1D",
    "deprived 2D",
    "deprived 3D",
    "deprived 4D"
]


deprivation_2011.head()

deprivation_2011.to_csv("Deprivation2011_cleaned.csv", index=False)


tenure_2011_matched = pd.read_csv('tenure_2011_cleaned.csv')
occupancy_2011_matched = pd.read_csv('occupancy_2011_cleaned.csv')
heating_2011 = pd.read_csv('heating2011.csv')
deprivation_2011 = pd.read_csv('Deprivation2011_cleaned.csv')

# Merging the DataFrames sequentially on both "geography" and "geography code"
merged_2011_df = pd.merge(tenure_2011_matched, occupancy_2011_matched,
                           on=["geography", "geography code"], how="outer")
merged_2011_df = pd.merge(merged_2011_df, heating_2011,
                           on=["geography", "geography code"], how="outer")
merged_2011_df = pd.merge(merged_2011_df, deprivation_2011,
                           on=["geography", "geography code"], how="outer")

merged_2011_df = merged_2011_df.drop(columns=['date_x', 'Rural Urban', 'date_y', 'Rural Urban','rural urban'])


merged_2011_df.head()


merged_2011_df.to_csv('merged_2011_data.csv', index=False)


merged_2011_df.isnull().sum()

tenure_2021_matched = pd.read_csv('tenure_2021_cleaned.csv')
occupancy_2021_matched = pd.read_csv('occupancy_2021_cleaned.csv')

merged_df = pd.merge(
    tenure_2021_matched,
    occupancy_2021_matched,
    on=["geography", "geography code"],
    how="inner"
)
merged_df.drop(columns=['Does not apply'], inplace=True)
merged_df.head()

merged_df.isnull().sum()

merged_df.to_csv('merged_2021_data.csv', index=False)


df_2011 = pd.read_csv('merged_2011_data.csv')
df_2021 = pd.read_csv('merged_2021_data.csv')

# Adding the date column
df_2011['date'] = 2011
df_2021['date'] = 2021

df_2011.to_csv('2011data_with_date.csv', index=False)
df_2021.to_csv('2021data_with_date.csv', index=False)




df_2011 = pd.read_csv("2011data_with_date.csv")
df_2021 = pd.read_csv("2021data_with_date.csv")

common_geography = df_2011[df_2011['geography'].isin(df_2021['geography'])]

df_combined = pd.concat([common_geography, df_2021[df_2021['geography'].isin(df_2011['geography'])]], ignore_index=True)


# Reorder columns to make 'date' the first column
columns = ['date'] + [col for col in df_combined.columns if col != 'date']
df_combined = df_combined[columns]

# Save the combined file
df_combined.to_csv("Final_merge.csv", index=False)

# Check the final output
df_combined.head()


df_combined.isnull().sum()

# Loading the merged dataset
df = pd.read_csv("Final_merge.csv")

# List of occupancy columns that need imputation
occupancy_columns = [
    "Occupancy rating of rooms: +1",
    "Occupancy rating of rooms: +2 or more",
    "Occupancy rating of rooms: -1",
    "Occupancy rating of rooms: -2 or less",
    "Occupancy rating of rooms: 0"
]

# List of numerical columns that need imputation using regression
numerical_columns = [
    "Central Heating: All categories: Type of central heating in household; measures: Value",
    "Central Heating: No central heating; measures: Value",
    "Central Heating: Gas central heating; measures: Value",
    "Central Heating: Electric (including storage heaters) central heating; measures: Value",
    "Central Heating: Oil central heating; measures: Value",
    "Central Heating: Solid fuel (for example wood, coal) central heating; measures: Value",
    "Central Heating: Other central heating; measures: Value",
    "Central Heating: Two or more types of central heating; measures: Value",
    "total households",
    "not deprived",
    "deprived 1D",
    "deprived 2D",
    "deprived 3D",
    "deprived 4D"
]

# List of categorical columns that need imputation using mode
categorical_columns = [
    "date", "geography", "geography code", "Lives rent free",
    "Owned: Owns outright", "Owned: Owns with a mortgage or loan",
    "Private rented: Other private rented", "Private rented: Private landlord or letting agency",
    "Shared ownership: Shared ownership", "Social rented: Other social rented",
    "Social rented: Rents from council or Local Authority"
]

# Imputation for categorical columns using mode
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        print(f"Imputing categorical column '{col}' with mode...")
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df[col] = mode_imputer.fit_transform(df[[col]])

# predictors for Bayesian Ridge Regression
predictors = [
    "Owned: Owns outright",
    "deprived 1D",
    "Central Heating: Gas central heating; measures: Value"
]

# Imputing missing values in the predictors
print("Imputing missing values in predictors...")
predictor_imputer = SimpleImputer(strategy='mean')
df[predictors] = predictor_imputer.fit_transform(df[predictors])

# Imputing missing values for numerical columns using Bayesian Ridge Regression
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        print(f"\nImputing numerical column '{col}' ({df[col].isnull().sum()} missing values)...")

        # Training data
        train_data = df.dropna(subset=[col] + predictors)
        X_train = train_data[predictors].values
        y_train = train_data[col].values

        # Fit Bayesian Ridge Regression model
        model = BayesianRidge()
        model.fit(X_train, y_train)

        # Predicting for rows with missing target values
        missing_mask = df[col].isnull()
        X_missing = df.loc[missing_mask, predictors].values
        imputed_values = model.predict(X_missing)

        # Fill missing values in the original DataFrame
        df.loc[missing_mask, col] = imputed_values

        print(f"Filled {missing_mask.sum()} missing values in '{col}'.")

df.to_csv("Final _all.csv", index=False)



df.isnull().sum()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from google.colab import files

df = pd.read_csv('Final _all.csv')

# Calculating derived features
df["%_private_rented"] = df["Private rented: Private landlord or letting agency"] / df["total households"]* 100
df["%_owned_outright"] = df["Owned: Owns outright"] / df["total households"]*100
df["%_deprived"] = (df["deprived 1D"] + df["deprived 2D"] + df["deprived 3D"] + df["deprived 4D"]) / df["total households"]* 100
df["%_no_central_heating"] = df["Central Heating: No central heating; measures: Value"] / df["total households"]* 100
df["%_overcrowded"] = (
    df["Occupancy rating of rooms: -1"] + df["Occupancy rating of rooms: -2 or less"]
) / df["total households"]*100

# Select and scale features
features = ["%_private_rented", "%_deprived", "%_overcrowded","%_no_central_heating"]
X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
pca = PCA(n_components=2)
df[["PC1", "PC2"]] = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
df[["tSNE1", "tSNE2"]] = tsne.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Interpret cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
cluster_centers["cluster"] = cluster_centers.index
print("Cluster Characteristics:\n", cluster_centers)

# Assign descriptive labels to clusters
def name_cluster(row):
    if row['cluster'] == 0:
        return "Low Deprivation & Renting"
    elif row['cluster'] == 1:
        return "High Overcrowding & Deprivation"
    elif row['cluster'] == 2:
        return "Moderate Everything"
    else:
        return "Unknown"

df["cluster_label"] = df.apply(name_cluster, axis=1)

# Saving for Tableau
df.to_csv("visualization_outputs.csv", index=False)



plt.figure(figsize=(20, 6))

# PCA Projection
plt.subplot(1, 3, 1)
scatter1 = plt.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="viridis", s=40, alpha=0.7, edgecolor='k')
plt.title("PCA Projection with Cluster Labels", fontsize=14)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=12)
cbar1 = plt.colorbar(scatter1)
cbar1.set_label('Cluster', fontsize=12)
cbar1.set_ticks([0, 1, 2])
cbar1.set_ticklabels([
    "Low Deprivation & Renting",
    "High Overcrowding & Deprivation",
    "Moderate Everything"
])

# t-SNE Projection
plt.subplot(1, 3, 2)
scatter2 = plt.scatter(df["tSNE1"], df["tSNE2"], c=df["cluster"], cmap="viridis", s=40, alpha=0.7, edgecolor='k')
plt.title("t-SNE Projection with Cluster Labels", fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
cbar2 = plt.colorbar(scatter2)
cbar2.set_label('Cluster', fontsize=12)
cbar2.set_ticks([0, 1, 2])
cbar2.set_ticklabels([
    "Low Deprivation & Renting",
    "High Overcrowding & Deprivation",
    "Moderate Everything"
])

# PCA Biplot
plt.subplot(1, 3, 3)
plt.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="viridis", s=20, alpha=0.3, edgecolor='k')
loadings = pca.components_.T
for i, feature in enumerate(features):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='red', alpha=0.9, head_width=0.1)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, color='red', fontsize=11)
plt.title("PCA Biplot (Feature Directions)", fontsize=14)
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)

plt.tight_layout()
plt.show()


df =pd.read_csv("visualization_outputs.csv")
df.head()

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge

df = pd.read_csv("visualization_outputs.csv")

# Extracting list of all unique regions
regions = df['geography'].unique()

# Key metrics to predict
target_features = [
    "%_private_rented",
    "%_deprived",
    "%_overcrowded",
    "%_no_central_heating"
]

# Creating dataframe for 2030 predictions
predictions_2030 = []

for region in regions:
    region_data = df[df['geography'] == region]

    if len(region_data) >= 1:
        pred_row = {
            'geography': region,
            'geography code': region_data['geography code'].iloc[0],
            'date': 2030
        }

        # Predicting each feature
        for feature in target_features:

            X = region_data[['date']].values
            y = region_data[feature].values

            if len(X) >= 2:
                # Training BayesianRidge model
                model = BayesianRidge()
                model.fit(X, y)

                # Predicting for 2030
                pred_value = model.predict([[2030]])[0]


                pred_row[feature] = pred_value
            elif len(X) == 1:
                # Calculating average yearly change across all regions
                all_regions_avg_change = 0
                regions_with_data = 0

                for other_region in regions:
                    other_data = df[df['geography'] == other_region]
                    if len(other_data) >= 2:
                        first_val = other_data.sort_values('date')[feature].iloc[0]
                        last_val = other_data.sort_values('date')[feature].iloc[-1]
                        first_year = other_data.sort_values('date')['date'].iloc[0]
                        last_year = other_data.sort_values('date')['date'].iloc[-1]

                        if first_val != 0:  # Avoid division by zero
                            yearly_change = (last_val - first_val) / (last_year - first_year)
                            all_regions_avg_change += yearly_change
                            regions_with_data += 1

                if regions_with_data > 0:
                    avg_yearly_change = all_regions_avg_change / regions_with_data
                    base_value = region_data[feature].iloc[0]
                    base_year = region_data['date'].iloc[0]
                    pred_value = base_value + avg_yearly_change * (2030 - base_year)
                else:

                    pred_value = region_data[feature].iloc[0]

                pred_row[feature] = pred_value

        # Add the row to the predictions
        predictions_2030.append(pred_row)

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions_2030)

# Copy non-predicted columns from the most recent data point for each region
for region in regions:
    region_data = df[df['geography'] == region]
    if not region_data.empty:
        latest_data = region_data.loc[region_data['date'].idxmax()]
        region_idx = predictions_df['geography'] == region

        if any(region_idx):
            # Copy all columns that aren't already in predictions_df
            for col in df.columns:
                if col not in predictions_df.columns and col != 'date':
                    predictions_df.loc[region_idx, col] = latest_data[col]

# Ensuring the columns are in a logical order
column_order = ['date', 'geography', 'geography code',"%_private_rented",
    "%_deprived",
    "%_overcrowded",
    "%_no_central_heating"]
predictions_df = predictions_df[column_order]

# Save the file for tableau
predictions_df.to_csv("regions_predictions_2030.csv", index=False)

print(f"Predictions for {len(predictions_df)} regions in 2030 have been saved to 'regions_predictions_2030.csv'")
