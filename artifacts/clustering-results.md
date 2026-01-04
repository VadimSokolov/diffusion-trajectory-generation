# Trip Clustering Analysis Results

## Overview
We performed K-Means clustering (K=4) on vehicle speed trajectories from the CMAP 2007 dataset. Features were extracted from 6367 trips, including average speed, maximum speed, idle time ratio, stops per km, and acceleration noise.

## Clusters Identified

| Cluster | Label | Avg Speed (m/s) | Max Speed (m/s) | Stops/km | Idle Ratio | Count | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Arterial/Suburban** | 15.6 | 22.6 | 0.59 | 2.0% | 2224 | Moderate speeds, regular stops. Likely suburban arterial roads. |
| **1** | **Highway/Interstate** | 22.2 | 30.8 | 0.12 | 0.5% | 1020 | High speeds, very few stops, low idle. Likely highway travel. |
| **2** | **Congested/City** | 13.9 | 21.8 | 1.29 | 4.4% | 636 | Low speeds, frequent stops, high idle time. Likely urban or congested traffic. |
| **3** | **Free-flow Arterial**| 16.7 | 22.7 | 0.28 | 1.0% | 2487 | Moderate speeds but fewer stops than Cluster 0. Likely free-flow major roads. |

## Visualization

### Cluster Centers (Heatmap)
![Cluster Centers](../fig/cluster/cluster_centers.png)

### PCA Projection
![PCA Projection](../fig/cluster/pca_projection.png)

### t-SNE Projection
t-SNE often reveals non-linear structures better than PCA. Even here, some transitions between clusters may appear continuous, reflecting the natural gradient of driving behaviors (e.g., from free-flow to congested).
![t-SNE Projection](../fig/cluster/tsne_projection.png)

### Feature Distributions
- **Average Speed**: Distinct separation between Highway (1) and Congested (2).
![Average Speed](../fig/cluster/boxplot_avg_speed_mps.png)

- **Stops per KM**: High variability in Congested (2) cluster.
![Stops per KM](../fig/cluster/boxplot_stops_per_km.png)

## Sample Trajectories

### Cluster 0 Samples
<p float="left">
  <img src="../fig/speed/speed_trip_23420.png" width="32%" />
  <img src="../fig/speed/speed_trip_29817.png" width="32%" />
  <img src="../fig/speed/speed_trip_28444.png" width="32%" />
  <img src="../fig/speed/speed_trip_11138.png" width="32%" />
  <img src="../fig/speed/speed_trip_15233.png" width="32%" />
  <img src="../fig/speed/speed_trip_16166.png" width="32%" />
</p>

### Cluster 1 Samples
<p float="left">
  <img src="../fig/speed/speed_trip_40632.png" width="32%" />
  <img src="../fig/speed/speed_trip_8475.png" width="32%" />
  <img src="../fig/speed/speed_trip_42575.png" width="32%" />
  <img src="../fig/speed/speed_trip_24998.png" width="32%" />
  <img src="../fig/speed/speed_trip_5051.png" width="32%" />
  <img src="../fig/speed/speed_trip_44930.png" width="32%" />
</p>

### Cluster 2 Samples
<p float="left">
  <img src="../fig/speed/speed_trip_40332.png" width="32%" />
  <img src="../fig/speed/speed_trip_23213.png" width="32%" />
  <img src="../fig/speed/speed_trip_13684.png" width="32%" />
  <img src="../fig/speed/speed_trip_21747.png" width="32%" />
  <img src="../fig/speed/speed_trip_7773.png" width="32%" />
  <img src="../fig/speed/speed_trip_8158.png" width="32%" />
</p>

### Cluster 3 Samples
<p float="left">
  <img src="../fig/speed/speed_trip_43182.png" width="32%" />
  <img src="../fig/speed/speed_trip_11903.png" width="32%" />
  <img src="../fig/speed/speed_trip_13431.png" width="32%" />
  <img src="../fig/speed/speed_trip_766.png" width="32%" />
  <img src="../fig/speed/speed_trip_43128.png" width="32%" />
  <img src="../fig/speed/speed_trip_16774.png" width="32%" />
</p>
