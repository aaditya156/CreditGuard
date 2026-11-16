"""Plotly-powered interactive visualizations for credit risk intelligence."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import FIGURES_DIR, TARGET_COL


class CreditRiskVisualizer:
    """Creates interactive Plotly figures for credit risk distribution and behavioral analysis."""

    def __init__(self, df: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None):
        self.df = df.copy()
        self.output_dir = Path(output_dir or FIGURES_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Corporate theme styling
        self.color_map = {0: "#2ca02c", 1: "#d62728"}  # Green = Non-default, Red = Default
        self.color_seq = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def _save_fig(self, fig: go.Figure, filename: str) -> Path:
        """Helper to save interactive HTML visualization."""
        html_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(html_path))
        return html_path

    def plot_target_distribution(self, save_name: Optional[str] = "target_distribution") -> go.Figure:
        """Plot dual donut chart and bar breakdown of loan_status."""
        counts = self.df[TARGET_COL].value_counts().reset_index()
        counts.columns = ["loan_status", "count"]
        counts["label"] = counts["loan_status"].map({0: "Non-Default (Fully Paid)", 1: "Default (Charged-off)"})
        counts["percent"] = (counts["count"] / counts["count"].sum()) * 100

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=["Portfolio Breakdown (%)", "Record Counts by Class"],
        )

        fig.add_trace(
            go.Pie(
                labels=counts["label"],
                values=counts["count"],
                hole=0.45,
                marker=dict(colors=["#2ca02c", "#d62728"]),
                textinfo="label+percent",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=counts["label"],
                y=counts["count"],
                marker_color=["#2ca02c", "#d62728"],
                text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts["count"], counts["percent"])],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text="CreditGuard: Target Variable Distribution (loan_status)",
            template="plotly_white",
            height=450,
        )

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_loan_grade_default_analysis(
        self, save_name: Optional[str] = "loan_grade_analysis"
    ) -> go.Figure:
        """Plot credit grade monotonic risk curve and borrower volume."""
        grade_stats = (
            self.df.groupby("loan_grade", observed=False)
            .agg(
                total_volume=("loan_amnt", "count"),
                default_rate=(TARGET_COL, "mean"),
                avg_int_rate=("loan_int_rate", "mean"),
            )
            .reset_index()
            .sort_values(by="loan_grade")
        )

        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
        )

        # Bar: Loan Volume
        fig.add_trace(
            go.Bar(
                x=grade_stats["loan_grade"],
                y=grade_stats["total_volume"],
                name="Borrower Volume",
                marker_color="#1f77b4",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        # Line: Default Rate
        fig.add_trace(
            go.Scatter(
                x=grade_stats["loan_grade"],
                y=grade_stats["default_rate"] * 100,
                name="Default Rate (%)",
                mode="lines+markers+text",
                text=[f"{r*100:.1f}%" for r in grade_stats["default_rate"]],
                textposition="top center",
                line=dict(color="#d62728", width=3),
                marker=dict(size=9),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Credit Risk & Volume by Loan Grade (Monotonic Risk Curve)",
            xaxis_title="Credit Rating Grade",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            height=480,
        )
        fig.update_yaxes(title_text="Number of Borrowers", secondary_y=False)
        fig.update_yaxes(title_text="Default Rate (%)", secondary_y=True, range=[0, 105])

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_risk_heatmap(
        self,
        row_feat: str = "loan_grade",
        col_feat: str = "loan_intent",
        save_name: Optional[str] = "risk_heatmap",
    ) -> go.Figure:
        """Plot default rate risk matrix heatmap across two dimensions."""
        pivot_df = pd.pivot_table(
            self.df,
            values=TARGET_COL,
            index=row_feat,
            columns=col_feat,
            aggfunc="mean",
            observed=False,
        ) * 100

        text_matrix = [[f"{val:.1f}%" if not np.isnan(val) else "N/A" for val in row] for row in pivot_df.values]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale="Reds",
                text=text_matrix,
                texttemplate="%{text}",
                colorbar=dict(title="Default Rate (%)"),
            )
        )

        fig.update_layout(
            title_text=f"Credit Risk Heatmap: Default Rate by {row_feat.replace('_', ' ').title()} vs {col_feat.replace('_', ' ').title()}",
            xaxis_title=col_feat.replace("_", " ").title(),
            yaxis_title=row_feat.replace("_", " ").title(),
            template="plotly_white",
            height=480,
        )

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_correlation_matrix(self, save_name: Optional[str] = "correlation_heatmap") -> go.Figure:
        """Plot Pearson correlation heatmap for numerical attributes."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr = self.df[num_cols].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title_text="Credit Risk Feature Correlation Matrix",
            template="plotly_white",
            height=600,
            width=750,
        )

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_financial_behavior_scatter(
        self, sample_size: int = 5000, save_name: Optional[str] = "financial_behavior_scatter"
    ) -> go.Figure:
        """Scatter plot of Interest Rate vs Loan Percent of Income colored by Default Status."""
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42).copy()
        sample_df["Default Status"] = sample_df[TARGET_COL].map({0: "Non-Default", 1: "Default"})

        fig = px.scatter(
            sample_df,
            x="loan_percent_income",
            y="loan_int_rate",
            color="Default Status",
            color_discrete_map={"Non-Default": "#2ca02c", "Default": "#d62728"},
            opacity=0.5,
            title="Financial Stress Frontier: Loan % of Income vs Interest Rate",
            labels={
                "loan_percent_income": "Loan Amount / Annual Income Ratio (Debt Burden)",
                "loan_int_rate": "Interest Rate (%)",
            },
            template="plotly_white",
            hover_data=["loan_amnt", "person_income", "loan_grade"],
        )

        # Add benchmark vertical stress lines
        fig.add_vline(x=0.35, line_width=2, line_dash="dash", line_color="orange", annotation_text="35% Debt Warning")
        fig.add_vline(x=0.50, line_width=2, line_dash="dash", line_color="red", annotation_text="50% Critical Stress")

        fig.update_layout(height=500)

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_iv_ranking(
        self, iv_df: pd.DataFrame, save_name: Optional[str] = "information_value_ranking"
    ) -> go.Figure:
        """Plot horizontal bar chart ranking features by Information Value (IV)."""
        sorted_df = iv_df.sort_values(by="information_value", ascending=True).copy()

        # Color mapping by predictive tier
        def get_color(iv):
            if iv < 0.02:
                return "#7f7f7f"
            elif iv < 0.10:
                return "#17becf"
            elif iv < 0.30:
                return "#1f77b4"
            elif iv < 0.50:
                return "#2ca02c"
            else:
                return "#ff7f0e"

        colors = [get_color(v) for v in sorted_df["information_value"]]

        fig = go.Figure(
            go.Bar(
                x=sorted_df["information_value"],
                y=sorted_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in sorted_df["information_value"]],
                textposition="outside",
            )
        )

        fig.add_vline(x=0.02, line_dash="dot", line_color="gray", annotation_text="Weak cutoff (0.02)")
        fig.add_vline(x=0.10, line_dash="dot", line_color="blue", annotation_text="Medium cutoff (0.10)")
        fig.add_vline(x=0.30, line_dash="dot", line_color="green", annotation_text="Strong cutoff (0.30)")

        fig.update_layout(
            title_text="Credit Risk Feature Predictive Power (Information Value / IV Ranking)",
            xaxis_title="Information Value (IV)",
            yaxis_title="Feature",
            template="plotly_white",
            height=450,
        )

        if save_name:
            self._save_fig(fig, save_name)
        return fig

    def plot_distributions_by_target(
        self, feature_col: str, save_name: Optional[str] = None
    ) -> go.Figure:
        """Plot violin/box distribution comparison of a continuous feature split by target."""
        plot_name = save_name or f"distribution_{feature_col}"
        temp = self.df.copy()
        temp["loan_status_label"] = temp[TARGET_COL].map({0: "Non-Default", 1: "Default"})

        fig = px.box(
            temp,
            x="loan_status_label",
            y=feature_col,
            color="loan_status_label",
            color_discrete_map={"Non-Default": "#2ca02c", "Default": "#d62728"},
            points="outliers",
            title=f"Distribution of {feature_col.replace('_', ' ').title()} by Default Outcome",
            labels={"loan_status_label": "Loan Outcome", feature_col: feature_col.replace("_", " ").title()},
            template="plotly_white",
        )

        fig.update_layout(height=450, showlegend=False)

        if plot_name:
            self._save_fig(fig, plot_name)
        return fig

