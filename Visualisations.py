import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class FeatureSpaceVisualizer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self.dataset_name = os.path.splitext(os.path.basename(config_path))[0]

        # Load configuration
        self.config = self._load_config()
        self.target_col = self.config["target_column"]
        self.id_col = self.config.get("id_column", "object_id")

        # Load data
        self.train_df, self.test_df = self._load_data()

        # Visualization settings
        self.output_dir = os.path.join("visualizations", self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.animation_steps = 36
        self.point_size = 6
        self.point_opacity = 0.8
        self.max_features_to_display = 5

    def _load_config(self) -> Dict:
        with open(self.config_path) as f:
            return json.load(f)

    def _load_data(self) -> tuple:
        train_path = os.path.join(self.config_dir, "Last_training.csv")
        test_path = os.path.join(self.config_dir, "Last_testing.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if self.id_col not in train_df.columns:
            train_df[self.id_col] = [f"train_{i}" for i in range(len(train_df))]
            test_df[self.id_col] = [f"test_{i}" for i in range(len(test_df))]

        train_df[self.target_col] = train_df[self.target_col].astype(str)
        test_df[self.target_col] = test_df[self.target_col].astype(str)

        return train_df, test_df

    def create_all_visualizations(self):
        print(f"Creating visualizations for {self.dataset_name}...")

        for set_name, df in [("train", self.train_df), ("test", self.test_df)]:
            print(f"\nProcessing {set_name} set ({len(df)} samples)...")

            # Create individual visualizations
            self._create_projection_plots(df, set_name)
            self._create_feature_distributions(df, set_name)
            self._create_correlation_matrix(df, set_name)
            self._create_parallel_coordinates(df, set_name)

        print(f"\nAll visualizations saved to: {self.output_dir}")

    def _create_projection_plots(self, df: pd.DataFrame, set_name: str):
        features = StandardScaler().fit_transform(
            df[
                [col for col in df.columns if col not in [self.target_col, self.id_col]]
            ].values
        )
        ids = df[self.id_col]
        labels = df[self.target_col]

        # Create projections
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(features)

        tsne = TSNE(n_components=3, perplexity=min(30, len(df) - 1), random_state=42)
        tsne_proj = tsne.fit_transform(features)

        # Create separate figures for each projection type
        self._create_single_projection(
            pca_proj,
            labels,
            ids,
            f"PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_):.2f})",
            set_name,
            "pca",
        )

        self._create_single_projection(
            tsne_proj, labels, ids, "t-SNE Projection", set_name, "tsne"
        )

    def _create_single_projection(
        self, projection, labels, ids, title, set_name, proj_type
    ):
        fig = go.Figure()

        unique_labels = sorted(labels.unique())
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(
                go.Scatter3d(
                    x=projection[mask, 0],
                    y=projection[mask, 1],
                    z=projection[mask, 2],
                    mode="markers",
                    name=label,
                    text=[f"ID: {id}<br>Class: {label}" for id in ids[mask]],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(
                        size=self.point_size,
                        opacity=self.point_opacity,
                        line=dict(width=0.5, color="DarkSlateGrey"),
                    ),
                )
            )

        # Camera animation setup
        frames = []
        for i in range(self.animation_steps):
            angle = 2 * np.pi * i / self.animation_steps
            eye_x = np.cos(angle) * 2
            eye_y = np.sin(angle) * 2

            frames.append(
                go.Frame(
                    layout=dict(scene_camera=dict(eye=dict(x=eye_x, y=eye_y, z=0.3)))
                )
            )

        fig.frames = frames

        fig.update_layout(
            title=f"{self.dataset_name} - {set_name} Set: {title}",
            scene=dict(
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.1)),
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            ),
            height=800,
            showlegend=True,
            hovermode="closest",
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play Rotation",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 50, "redraw": True}}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
        )

        fig.write_html(os.path.join(self.output_dir, f"3d_{proj_type}_{set_name}.html"))

    def _create_feature_distributions(self, df: pd.DataFrame, set_name: str):
        feature_cols = [
            col for col in df.columns if col not in [self.target_col, self.id_col]
        ]
        features_to_display = feature_cols[: self.max_features_to_display]

        fig = make_subplots(
            rows=len(features_to_display),
            cols=len(features_to_display),
            subplot_titles=[
                f"{f1} vs {f2}"
                for f1 in features_to_display
                for f2 in features_to_display
            ],
        )

        for i, f1 in enumerate(features_to_display):
            for j, f2 in enumerate(features_to_display):
                if i == j:
                    # Diagonal - show distribution
                    for label in df[self.target_col].unique():
                        mask = df[self.target_col] == label
                        fig.add_trace(
                            go.Violin(
                                x=df.loc[mask, f1],
                                name=f"{label}",
                                legendgroup=label,
                                showlegend=(i == 0 and j == 0),
                                line_color=px.colors.qualitative.Plotly[
                                    list(df[self.target_col].unique()).index(label)
                                    % len(px.colors.qualitative.Plotly)
                                ],
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                else:
                    # Off-diagonal - show scatter plot
                    for label in df[self.target_col].unique():
                        mask = df[self.target_col] == label
                        fig.add_trace(
                            go.Scatter(
                                x=df.loc[mask, f1],
                                y=df.loc[mask, f2],
                                mode="markers",
                                name=label,
                                legendgroup=label,
                                showlegend=False,
                                marker=dict(
                                    color=px.colors.qualitative.Plotly[
                                        list(df[self.target_col].unique()).index(label)
                                        % len(px.colors.qualitative.Plotly)
                                    ],
                                    size=5,
                                    opacity=0.7,
                                ),
                            ),
                            row=i + 1,
                            col=j + 1,
                        )

        fig.update_layout(
            title=f"{self.dataset_name} - {set_name} Set: Feature Space Distributions",
            height=200 * len(features_to_display),
            width=200 * len(features_to_display),
            showlegend=True,
        )

        fig.write_html(
            os.path.join(self.output_dir, f"feature_distributions_{set_name}.html")
        )

    def _create_correlation_matrix(self, df: pd.DataFrame, set_name: str):
        feature_cols = [
            col for col in df.columns if col not in [self.target_col, self.id_col]
        ]
        corr = df[feature_cols].corr()

        fig = px.imshow(
            corr,
            title=f"Feature Correlation Matrix ({set_name} set)",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=1000, width=1000)
        fig.write_html(os.path.join(self.output_dir, f"correlation_{set_name}.html"))

    def _create_parallel_coordinates(self, df: pd.DataFrame, set_name: str):
        feature_cols = [
            col for col in df.columns if col not in [self.target_col, self.id_col]
        ]
        sample_features = feature_cols[:10]  # Limit to 10 features

        # Create a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()

        # Convert categorical labels to numerical codes for coloring
        plot_df["color_code"] = pd.factorize(plot_df[self.target_col])[0]

        fig = px.parallel_coordinates(
            plot_df,
            dimensions=sample_features + [self.target_col],
            color="color_code",
            color_continuous_scale=px.colors.qualitative.Plotly,
            title=f"Parallel Coordinates Plot ({set_name} set)",
            labels={"color_code": self.target_col},
        )

        # Update the color axis to show the original labels
        unique_labels = sorted(plot_df[self.target_col].unique())
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=self.target_col,
                tickvals=list(range(len(unique_labels))),
                ticktext=unique_labels,
            ),
            height=800,
        )

        fig.write_html(
            os.path.join(self.output_dir, f"parallel_coords_{set_name}.html")
        )


def main():
    config_path = input("Enter path to config file: ").strip()
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    try:
        visualizer = FeatureSpaceVisualizer(config_path)
        visualizer.create_all_visualizations()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
