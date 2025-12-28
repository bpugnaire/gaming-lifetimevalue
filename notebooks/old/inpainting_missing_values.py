import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import plotly.express as px
    from datetime import datetime, timedelta
    import numpy as np
    import plotly.graph_objects as go
    return Path, go, pl


@app.cell
def _(Path):
    data_path = Path("data")
    return (data_path,)


@app.cell
def _(data_path, pl):
    train_data = pl.read_parquet(data_path / "train_samples.parquet")
    return (train_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Looking for linearity of revenue growth accross time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our dataset is structured in cumulative features at different horizons (d0, d3, d7, d14, d30, d60, d90, d120).
    So the closer we are to the pipeline run date, the less data we have in terms of horizon.

    Our dataset has roughly 3M rows but nearly half the d120 values are missing. A first interesting approach would be to try to fill the missing values (or at least some) to achieve 2 goals:
    - have more training data
    - have fresher data for users that installed recently (that will likely be more representative of current user behavior)

    So we can try to look for a simple correlation / linearity along time for our key revenue features.
    """)
    return


@app.cell
def _(pl, train_data):
    no_rev_users = train_data.filter(pl.col("d120_rev") == 0)
    rev_users = train_data.filter(pl.col("d120_rev") > 0)
    return (rev_users,)


@app.cell
def _(pl, rev_users):
    rev_users_segmented = rev_users.with_columns(
        top_rev_1pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
    )
    return (rev_users_segmented,)


@app.cell
def _(pl, rev_users_segmented):
    df_with_cohorts = rev_users_segmented.with_columns(
        cohort = pl.when(pl.col("top_rev_1pct")).then(pl.lit("Top 1%"))
                .when(pl.col("top_rev_5pct")).then(pl.lit("Top 5%"))
                .when(pl.col("top_rev_20pct")).then(pl.lit("Top 20%"))
                .otherwise(pl.lit("Others"))
    )
    return (df_with_cohorts,)


@app.cell
def _(df_with_cohorts, pl):
    postfix = "rev"
    steps = [0, 14, 30, 60, 90, 120]
    cols = [f"d{s}_{postfix}" for s in steps]

    stats_df = df_with_cohorts.group_by('cohort').agg(
        pl.col(cols).mean().name.suffix("_mean"),
        pl.col(cols).std().name.suffix("_std")
    )


    stats_df.sort("cohort")
    return postfix, stats_df, steps


@app.cell
def _(go, postfix, stats_df, steps):
    def plot_simple_cum_rev():
        fig = go.Figure()

        for row in stats_df.to_dicts():
            name = row["cohort"]
            y_values = [row[f"d{s}_{postfix}_mean"] for s in steps]
    
            fig.add_trace(go.Scatter(
                x=steps, 
                y=y_values, 
                mode='lines+markers', 
                name=name
            ))
    
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Days Since Install",
            yaxis_title=f"Cumulative {postfix.upper()}",
            hovermode="x unified"
        )
    
        fig.show()
    plot_simple_cum_rev()
    return


@app.cell
def _(df_with_cohorts, pl):
    multipliers = (
        df_with_cohorts
        .filter(pl.col("d120_rev").is_not_null())
        .group_by("cohort")
        .agg([
            (pl.col("d120_rev").mean() / pl.col("d30_rev").mean()).alias("m30_to_d120"),
            (pl.col("d120_rev").mean() / pl.col("d60_rev").mean()).alias("m60_to_d120"),
            (pl.col("d120_rev").mean() / pl.col("d90_rev").mean()).alias("m90_to_d120"),
            pl.len().alias("sample_size")
        ])
        .sort("m30_to_d120", descending=True)
    )

    multipliers
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
