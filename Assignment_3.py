import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (mean_absolute_error,mean_absolute_percentage_error)


# Load the 2021 training dataset
data_df = pd.read_csv("dataset_exercise_5_clustering_highway_traffic (1).csv",sep=";")

print(data_df.head())
print("\nDataset shape:", data_df.shape)
print("\nColumns:")
print(data_df.columns)

# Sort observations by date and 5-minute interval
data_df = data_df.sort_values(["Date", "Interval_5"]).reset_index(drop=True)

# Get all unique days
days = np.sort(data_df["Date"].unique())

# Number of unique days
ndays = len(days)

# Number of 5-minute intervals in 24 hours
nintvals = 288

print("Number of days:", ndays)
print("Intervals per day:", nintvals)

# Create empty day x interval matrix
vectorized_day_dataset = np.full((ndays, nintvals), np.nan)

# Fill the matrix with traffic flow
for i, day in enumerate(days):

    day_data = data_df[data_df["Date"] == day]

    intervals = day_data["Interval_5"].to_numpy(dtype=int)
    flows = day_data["flow"].to_numpy(dtype=float)

    vectorized_day_dataset[i, intervals] = flows

print("Shape of vectorized dataset:", vectorized_day_dataset.shape)

# Count all missing 5-minute observations
number_of_nans = np.sum(np.isnan(vectorized_day_dataset))

# Percentage/rate of missing observations
rate_of_nans = number_of_nans / (ndays * nintvals)

# Count missing observations for every day
nans_per_day = np.sum(np.isnan(vectorized_day_dataset), axis=1)

# Number of affected days
days_with_missing = np.sum(nans_per_day > 0)

print("Number of NaNs:", number_of_nans)
print("Rate of NaNs:", rate_of_nans)
print("Number of days with missing values:", days_with_missing)

# Convert intervals to hours
hours = np.arange(nintvals) * 5 / 60


# Keep only days without missing observations
complete_day_mask = nans_per_day == 0

vectorized_day_dataset_no_nans = vectorized_day_dataset[complete_day_mask]

days_not_nans = days[complete_day_mask]

print("Original number of days:", len(days))
print("Removed incomplete days:", np.sum(~complete_day_mask))
print("Final number of complete days:", len(days_not_nans))
print("Final dataset shape:", vectorized_day_dataset_no_nans.shape)

X = vectorized_day_dataset_no_nans
print("Clustering dataset:", X.shape)

# Convert valid dates to datetime
dates_datetime = pd.to_datetime(
    days_not_nans.astype(str),
    format="%Y%m%d"
)

# Monday = 0, Sunday = 6
weekday_number = dates_datetime.weekday

# Separate weekdays and weekends
weekday_mask = weekday_number < 5
weekend_mask = weekday_number >= 5

weekday_mean = X[weekday_mask].mean(axis=0)
weekend_mean = X[weekend_mask].mean(axis=0)

plt.figure(figsize=(13, 6))
plt.plot(hours, weekday_mean, label="Weekday average", linewidth=2)
plt.plot(hours, weekend_mean, label="Weekend average",linewidth=2)
plt.xlabel("Time of day [hours]")
plt.ylabel("Average traffic flow")
plt.title("Average Weekday and Weekend Traffic Profiles")
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 2))
plt.legend()
plt.show()



# Range of cluster numbers to investigate
k_values = range(2, 13)

# Store results
kmeans_results = []

for k in k_values:

    # Create K-means model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    # Fit model and obtain cluster labels
    labels = kmeans.fit_predict(X)

    # Calculate internal evaluation metrics
    silhouette = silhouette_score(X, labels)

    davies_bouldin = davies_bouldin_score(X, labels)

    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Inertia is also stored for the elbow method
    inertia = kmeans.inertia_

    # Store results
    kmeans_results.append({
        "n_clusters": k,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "inertia": inertia
    })


# Convert results into DataFrame
kmeans_results_df = pd.DataFrame(kmeans_results)

print(kmeans_results_df.round(3))

plt.figure(figsize=(8, 5))
plt.plot(kmeans_results_df["n_clusters"],kmeans_results_df["silhouette"],marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("K-means: Silhouette Score")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(kmeans_results_df["n_clusters"],kmeans_results_df["davies_bouldin"],marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("K-means: Davies-Bouldin Index")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(kmeans_results_df["n_clusters"],kmeans_results_df["calinski_harabasz"], marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.title("K-means: Calinski-Harabasz Score")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(kmeans_results_df["n_clusters"],kmeans_results_df["inertia"],marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("K-means Elbow Method")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.show()

best_silhouette = kmeans_results_df.loc[kmeans_results_df["silhouette"].idxmax()]
best_db = kmeans_results_df.loc[kmeans_results_df["davies_bouldin"].idxmin()]
best_ch = kmeans_results_df.loc[kmeans_results_df["calinski_harabasz"].idxmax()]

print("Best Silhouette:",int(best_silhouette["n_clusters"]),"clusters -",round(best_silhouette["silhouette"], 3))
print("Best Davies-Bouldin:",int(best_db["n_clusters"]),"clusters -",round(best_db["davies_bouldin"], 3))
print("Best Calinski-Harabasz:",int(best_ch["n_clusters"]),"clusters -",round(best_ch["calinski_harabasz"], 3))

for k in [2, 5]:

    model = KMeans(n_clusters=k,random_state=42,n_init=20)

    labels = model.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)

    print(f"\nK-means with {k} clusters")

    for cluster, count in zip(unique, counts):
        print(
            f"Cluster {cluster}: {count} days "
            f"({count / len(X) * 100:.1f}%)"
        )


# Temporary candidate for visual inspection
k = 5

kmeans_5 = KMeans(n_clusters=k,random_state=42,n_init=20)

kmeans_5_labels = kmeans_5.fit_predict(X)

kmeans_5_centroids = kmeans_5.cluster_centers_

plt.figure(figsize=(13, 7))

for cluster in range(k):

    plt.plot(hours,kmeans_5_centroids[cluster],linewidth=2,label=f"Cluster {cluster}")

plt.xlabel("Time of day [hours]")
plt.ylabel("Traffic flow")
plt.title("Representative Daily Traffic Profiles ""- K-means, k = 5")
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 2))
plt.legend()
plt.grid(alpha=0.2)
plt.show()

cluster_analysis = pd.DataFrame({"Date": dates_datetime,"Cluster": kmeans_5_labels})
cluster_analysis["Day_of_week"] = (cluster_analysis["Date"].dt.day_name())
cluster_analysis["Day_type"] = np.where(cluster_analysis["Date"].dt.weekday < 5,"Weekday","Weekend")
cluster_summary = pd.crosstab(cluster_analysis["Cluster"],cluster_analysis["Day_type"])
print(cluster_summary)

linkage_methods = ["ward", "complete", "average", "single"]
cluster_range = range(2, 13)

agglomerative_results = []

for linkage in linkage_methods:

    for k in cluster_range:

        model = AgglomerativeClustering(n_clusters=k,linkage=linkage)

        labels = model.fit_predict(X)

        # Internal evaluation
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)

        # Cluster sizes
        cluster_sizes = np.bincount(labels)

        agglomerative_results.append({
            "linkage": linkage,
            "n_clusters": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "smallest_cluster": cluster_sizes.min(),
            "largest_cluster": cluster_sizes.max()
        })


agglomerative_results_df = pd.DataFrame(agglomerative_results)
print(agglomerative_results_df.round(3))

best_agg_silhouette = (agglomerative_results_df.loc[agglomerative_results_df["silhouette"].idxmax()])
best_agg_db = (agglomerative_results_df.loc[agglomerative_results_df["davies_bouldin"].idxmin()])
best_agg_ch = (agglomerative_results_df.loc[agglomerative_results_df["calinski_harabasz"].idxmax()])

print("\nBest Silhouette:")
print(best_agg_silhouette)
print("\nBest Davies-Bouldin:")
print(best_agg_db)
print("\nBest Calinski-Harabasz:")
print(best_agg_ch)


# Number of neighbours used for the distance analysis
n_neighbors = 10

neighbors = NearestNeighbors(n_neighbors=n_neighbors)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Distance to the kth nearest neighbour
k_distances = np.sort(distances[:, n_neighbors - 1])

plt.figure(figsize=(9, 5))
plt.plot(k_distances)
plt.xlabel("Days sorted by distance")
plt.ylabel(f"Distance to {n_neighbors}th nearest neighbour")
plt.title("DBSCAN k-distance Plot")
plt.grid(alpha=0.3)
plt.show()

eps_values = np.arange(400, 701, 50)
min_samples_values = [3, 4, 5, 6, 8, 10]

dbscan_results = []

for eps in eps_values:

    for min_samples in min_samples_values:

        dbscan = DBSCAN(eps=eps,min_samples=min_samples,metric="euclidean")
        labels = dbscan.fit_predict(X)

        # DBSCAN labels noise as -1
        non_noise_mask = labels != -1

        # Unique clusters, excluding noise
        cluster_labels = np.unique(labels[non_noise_mask])
        n_clusters = len(cluster_labels)
        n_noise = np.sum(labels == -1)
        noise_percentage = (n_noise / len(X) * 100)

        # Internal metrics require at least 2 clusters
        if (n_clusters >= 2 and np.sum(non_noise_mask) > n_clusters):

            X_clustered = X[non_noise_mask]
            labels_clustered = labels[non_noise_mask]

            silhouette = silhouette_score(X_clustered,labels_clustered)
            davies_bouldin = davies_bouldin_score(X_clustered,labels_clustered)
            calinski_harabasz = calinski_harabasz_score(X_clustered,labels_clustered)

            cluster_sizes = [np.sum(labels == cluster)for cluster in cluster_labels]

            smallest_cluster = min(cluster_sizes)
            largest_cluster = max(cluster_sizes)

        else:

            silhouette = np.nan
            davies_bouldin = np.nan
            calinski_harabasz = np.nan
            smallest_cluster = np.nan
            largest_cluster = np.nan

        dbscan_results.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percentage": noise_percentage,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "smallest_cluster": smallest_cluster,
            "largest_cluster": largest_cluster
        })


dbscan_results_df = pd.DataFrame(dbscan_results)
print(dbscan_results_df.dropna().sort_values("silhouette", ascending=False).head(15).round(3))


plt.figure(figsize=(10, 6))

for min_samples in min_samples_values:

    subset = dbscan_results_df[
        dbscan_results_df["min_samples"]== min_samples]

    plt.plot(subset["eps"],subset["noise_percentage"],marker="o",label=f"min_samples={min_samples}")

plt.xlabel("eps")
plt.ylabel("Noise observations [%]")
plt.title("DBSCAN: Percentage of Days Classified as Noise")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


covariance_types = [
    "full",
    "tied",
    "diag",
    "spherical"
]

component_range = range(2, 13)

gmm_results = []

for covariance_type in covariance_types:

    for k in component_range:

        gmm = GaussianMixture(n_components=k,covariance_type=covariance_type,random_state=42,n_init=3,max_iter=300,reg_covar=1e-3)

        # Fit model and predict cluster labels
        labels = gmm.fit_predict(X)

        # Internal evaluation
        silhouette = silhouette_score(X,labels)

        davies_bouldin = davies_bouldin_score(X,labels)

        calinski_harabasz = calinski_harabasz_score(X,labels)

        # GMM-specific evaluation
        bic = gmm.bic(X)

        # Cluster sizes
        cluster_sizes = np.bincount(labels,minlength=k)

        gmm_results.append({
            "covariance_type": covariance_type,
            "n_components": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "BIC": bic,
            "smallest_cluster": cluster_sizes.min(),
            "largest_cluster": cluster_sizes.max()})


gmm_results_df = pd.DataFrame(gmm_results)
print(gmm_results_df.round(3))


best_gmm_silhouette = (gmm_results_df.loc[gmm_results_df["silhouette"].idxmax()])
best_gmm_db = (gmm_results_df.loc[gmm_results_df["davies_bouldin"].idxmin()])
best_gmm_ch = (gmm_results_df.loc[gmm_results_df["calinski_harabasz"].idxmax()])

print("\nBest Silhouette:")
print(best_gmm_silhouette)
print("\nBest Davies-Bouldin:")
print(best_gmm_db)
print("\nBest Calinski-Harabasz:")
print(best_gmm_ch)


plt.figure(figsize=(10, 6))

for covariance_type in covariance_types:

    subset = gmm_results_df[gmm_results_df["covariance_type"]== covariance_type]

    plt.plot(subset["n_components"],subset["BIC"],marker="o",label=covariance_type)

plt.xlabel("Number of components")
plt.ylabel("BIC")
plt.title("GMM: Bayesian Information Criterion")
plt.xticks(component_range)
plt.legend()
plt.grid(alpha=0.3)
plt.show()





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load independent 2022 evaluation dataset
evaluation_df = pd.read_csv("evaluation_dataset_exercise_5_clustering_highway_traffic.csv",sep=";")
evaluation_df = evaluation_df.sort_values(["Date", "Interval_5"]).reset_index(drop=True)
evaluation_days = np.sort(evaluation_df["Date"].unique())
n_eval_days = len(evaluation_days)

# Create day x interval matrix
evaluation_matrix = np.full((n_eval_days, 288),np.nan)

for i, day in enumerate(evaluation_days):

    day_data = evaluation_df[evaluation_df["Date"] == day]
    intervals = day_data["Interval_5"].to_numpy(dtype=int)
    flows = day_data["flow"].to_numpy(dtype=float)
    evaluation_matrix[i,intervals] = flows


# Find incomplete days
eval_nans_per_day = np.sum(np.isnan(evaluation_matrix),axis=1)
complete_eval_mask = (eval_nans_per_day == 0)
X_eval = evaluation_matrix[complete_eval_mask]
evaluation_days_complete = evaluation_days[complete_eval_mask]


print("Original evaluation days:",len(evaluation_days))
print("Incomplete days removed:",np.sum(~complete_eval_mask))
print("Complete evaluation days:",len(evaluation_days_complete))
print("Evaluation dataset shape:",X_eval.shape)


def evaluate_centroid_prediction(X_eval,centroids,history=5):

    actual_values = []
    predicted_values = []

    # Loop through evaluation days
    for day in X_eval:

        # Start after the first 5 observations
        for t in range(history, 288):

            # Previous 25 minutes
            observed_history = day[t-history:t]

            # Same time window for every centroid
            centroid_history = centroids[:,t-history:t]

            # Euclidean distance to each centroid
            distances = np.linalg.norm(centroid_history- observed_history,axis=1)

            # Select closest representative day-type
            closest_cluster = np.argmin(distances)
            
            # Predict next 5-minute flow
            prediction = centroids[closest_cluster,t]

            actual_values.append(day[t])

            predicted_values.append(prediction)


    actual_values = np.array(actual_values)

    predicted_values = np.array(predicted_values)


    # External evaluation
    mae = mean_absolute_error(actual_values,predicted_values)

    mape = (mean_absolute_percentage_error(actual_values,predicted_values)* 100)

    return mae, mape


kmeans_external_results = []

for k in range(2, 13):

    model = KMeans(n_clusters=k,random_state=42,n_init=20)

    model.fit(X)

    centroids = model.cluster_centers_

    mae, mape = evaluate_centroid_prediction(X_eval,centroids)

    kmeans_external_results.append({"method": "K-means","n_clusters": k,"MAE": mae,"MAPE": mape})


kmeans_external_df = pd.DataFrame(kmeans_external_results)

print(kmeans_external_df.round(3))


agglomerative_external_results = []
for k in range(2, 13):

    model = AgglomerativeClustering(n_clusters=k,linkage="ward")

    labels = model.fit_predict(X)

    # Agglomerative has no built-in centroids,
    # so calculate the average profile
    centroids = np.array([X[labels == cluster].mean(axis=0)for cluster in np.unique(labels)])

    mae, mape = evaluate_centroid_prediction(X_eval,centroids)

    agglomerative_external_results.append({"method": "Agglomerative Ward","n_clusters": k,"MAE": mae,"MAPE": mape})


agglomerative_external_df = pd.DataFrame(agglomerative_external_results)

print(agglomerative_external_df.round(3))


dbscan_external_results = []
eps_values = np.arange(400,701,50)
min_samples_values = [3, 4, 5, 6, 8, 10]

for eps in eps_values:

    for min_samples in min_samples_values:

        model = DBSCAN(eps=eps,min_samples=min_samples)

        labels = model.fit_predict(X)

        cluster_ids = np.unique(labels[labels != -1])

        # Need at least two actual clusters
        if len(cluster_ids) < 2:
            continue

        centroids = np.array([X[labels == cluster].mean(axis=0)for cluster in cluster_ids])

        mae, mape = evaluate_centroid_prediction(X_eval,centroids)

        noise_percentage = (np.sum(labels == -1)/ len(labels)* 100)

        dbscan_external_results.append({
            "method": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": len(cluster_ids),
            "noise_percentage": noise_percentage,
            "MAE": mae,
            "MAPE": mape})


dbscan_external_df = pd.DataFrame(dbscan_external_results)

print(dbscan_external_df.sort_values("MAPE").round(3))


gmm_external_results = []

covariance_types = ["full","tied","diag","spherical"]


for covariance_type in covariance_types:

    for k in range(2, 13):

        model = GaussianMixture(n_components=k,covariance_type=covariance_type,random_state=42,n_init=3,max_iter=300,reg_covar=1e-3)

        model.fit(X)

        # GMM component means are the centroids
        centroids = model.means_

        mae, mape = evaluate_centroid_prediction(X_eval,centroids)

        gmm_external_results.append({
            "method": "GMM",
            "covariance_type": covariance_type,
            "n_components": k,
            "MAE": mae,
            "MAPE": mape})


gmm_external_df = pd.DataFrame(gmm_external_results)

print(gmm_external_df.sort_values("MAPE").round(3))


# FINAL SELECTED CLUSTERING

final_model = KMeans(n_clusters=5,random_state=42,n_init=20)
final_labels = final_model.fit_predict(X)
final_centroids = final_model.cluster_centers_
print("Final model: K-means with 5 clusters")

unique_clusters, cluster_counts = np.unique(final_labels,return_counts=True)

print("\nFinal cluster sizes:")

for cluster, count in zip(unique_clusters,cluster_counts):

    print(f"Cluster {cluster}: "f"{count} days "f"({count / len(X) * 100:.1f}%)")


final_analysis = pd.DataFrame({"Date": dates_datetime,"Cluster": final_labels})
final_analysis["Day_type"] = np.where(final_analysis["Date"].dt.weekday < 5,"Weekday","Weekend")
final_daytype_summary = pd.crosstab(final_analysis["Cluster"],final_analysis["Day_type"])

print(final_daytype_summary)




#FINAL OUTPUTS


# K-MEANS: k = 2 and k = 5
kmeans_2_internal = kmeans_results_df[kmeans_results_df["n_clusters"] == 2].iloc[0]
kmeans_2_external = kmeans_external_df[kmeans_external_df["n_clusters"] == 2].iloc[0]

kmeans_5_internal = kmeans_results_df[kmeans_results_df["n_clusters"] == 5].iloc[0]
kmeans_5_external = kmeans_external_df[kmeans_external_df["n_clusters"] == 5].iloc[0]



# AGGLOMERATIVE: Ward, k = 5
agg_5_internal = agglomerative_results_df[(agglomerative_results_df["linkage"] == "ward")&(agglomerative_results_df["n_clusters"] == 5)].iloc[0]
agg_5_external = agglomerative_external_df[agglomerative_external_df["n_clusters"] == 5].iloc[0]



# BEST EXTERNAL DBSCAN
best_dbscan_external = dbscan_external_df.loc[dbscan_external_df["MAPE"].idxmin()]
best_dbscan_internal = dbscan_results_df[(dbscan_results_df["eps"] == best_dbscan_external["eps"])&(dbscan_results_df["min_samples"]== best_dbscan_external["min_samples"])].iloc[0]



# BEST EXTERNAL GMM
best_gmm_external = gmm_external_df.loc[gmm_external_df["MAPE"].idxmin()]
best_gmm_internal = gmm_results_df[(gmm_results_df["covariance_type"]== best_gmm_external["covariance_type"])&(gmm_results_df["n_components"]== best_gmm_external["n_components"])].iloc[0]



# FINAL COMPARISON TABLE
final_comparison = pd.DataFrame([
    {"Method": "K-means","Parameters": "k=2","Silhouette": kmeans_2_internal["silhouette"],"Davies-Bouldin": kmeans_2_internal["davies_bouldin"],"Calinski-Harabasz": kmeans_2_internal["calinski_harabasz"],"MAE": kmeans_2_external["MAE"],"MAPE": kmeans_2_external["MAPE"]},
    {"Method": "K-means","Parameters": "k=5","Silhouette": kmeans_5_internal["silhouette"],"Davies-Bouldin": kmeans_5_internal["davies_bouldin"],"Calinski-Harabasz": kmeans_5_internal["calinski_harabasz"],"MAE": kmeans_5_external["MAE"],"MAPE": kmeans_5_external["MAPE"]},
    { "Method": "Agglomerative", "Parameters": "Ward, k=5", "Silhouette": agg_5_internal["silhouette"], "Davies-Bouldin": agg_5_internal["davies_bouldin"], "Calinski-Harabasz": agg_5_internal["calinski_harabasz"], "MAE": agg_5_external["MAE"],"MAPE": agg_5_external["MAPE"]},
    {"Method": "DBSCAN","Parameters": (f"eps={best_dbscan_external['eps']}, "f"min_samples={int(best_dbscan_external['min_samples'])}"),"Silhouette": best_dbscan_internal["silhouette"],"Davies-Bouldin": best_dbscan_internal["davies_bouldin"],"Calinski-Harabasz": best_dbscan_internal["calinski_harabasz"],"MAE": best_dbscan_external["MAE"],"MAPE": best_dbscan_external["MAPE"]},
    {"Method": "GMM","Parameters": (f"{best_gmm_external['covariance_type']}, "f"{int(best_gmm_external['n_components'])} components"),"Silhouette": best_gmm_internal["silhouette"],"Davies-Bouldin": best_gmm_internal["davies_bouldin"],"Calinski-Harabasz": best_gmm_internal["calinski_harabasz"],"MAE": best_gmm_external["MAE"],"MAPE": best_gmm_external["MAPE"]}])

print(final_comparison.round(3))

# Save results
final_comparison.to_csv("final_clustering_comparison.csv",index=False)


plt.figure(figsize=(9, 5))
plt.plot(kmeans_external_df["n_clusters"],kmeans_external_df["MAPE"],marker="o")
plt.axvline(x=5,linestyle="--",label="Selected: k=5")
plt.xlabel("Number of clusters")
plt.ylabel("MAPE [%]")
plt.title("K-means External Evaluation on 2022 Data")
plt.xticks(range(2, 13))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_external_evaluation.png",dpi=300,bbox_inches="tight")
plt.show()


plt.figure(figsize=(13, 7))

for cluster in range(5):

    cluster_size = np.sum(final_labels == cluster)

    plt.plot(hours,final_centroids[cluster],linewidth=2,label=(f"Cluster {cluster} "f"(n={cluster_size})"))

plt.xlabel("Time of day [hours]")
plt.ylabel("Traffic flow")
plt.title("Representative Traffic Day-Types ""- K-means, k=5")
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 2))
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("representative_day_types_kmeans_5.png",dpi=300,bbox_inches="tight")
plt.show()


final_analysis = pd.DataFrame({"Date": dates_datetime,"Cluster": final_labels})
final_analysis["Day_type"] = np.where(final_analysis["Date"].dt.weekday < 5,"Weekday","Weekend")

day_type_table = pd.crosstab(final_analysis["Cluster"],final_analysis["Day_type"])
day_type_table["Total"] = (day_type_table.sum(axis=1))

print(day_type_table)

day_type_table.to_csv("cluster_day_type_composition.csv")