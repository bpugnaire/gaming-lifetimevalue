import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initial Data Exploration
    """)
    return


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    from skrub import TableReport
    from datetime import datetime, timedelta
    import numpy as np
    import plotly.graph_objects as go
    return Path, TableReport, datetime, pl, sns, timedelta


@app.cell
def _(Path):
    data_path = Path("data/raw/")
    return (data_path,)


@app.cell
def _(data_path, pl):
    train_data = pl.read_parquet(data_path / "train_samples.parquet")
    return (train_data,)


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell
def _(train_data):
    train_data.drop(['__index_level_0__'])
    return


@app.cell
def _(train_data):
    train_data.shape
    return


@app.cell
def _(train_data):
    train_data.n_unique("user_id")
    return


@app.cell
def _(train_data):
    train_data.null_count()
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell
def _(train_data):
    train_data.columns
    return


@app.cell
def _(train_data):
    user_install_features = train_data.columns[1:14]
    user_install_features
    return (user_install_features,)


@app.cell
def _(train_data):
    game_engagement_features = train_data.columns[14:]
    game_engagement_features
    return (game_engagement_features,)


@app.cell
def _():
    cum_features_horizons = [f'd{dx}' for dx in [0, 3, 7, 14, 30, 60, 90, 120]]
    cum_features_horizons
    return (cum_features_horizons,)


@app.cell
def _(cum_features_horizons, game_engagement_features):
    d0_columns = [col for col in game_engagement_features if cum_features_horizons[0] in col]
    d30_columns = [col for col in game_engagement_features if cum_features_horizons[4] in col]
    d120_columns = [col for col in game_engagement_features if cum_features_horizons[7] in col]

    d0_columns
    return d0_columns, d120_columns, d30_columns


@app.cell
def _(train_data):
    train_data.select("d120_rev").describe()
    return


@app.cell
def _(sns, train_data):
    sns.histplot(train_data, x="d120_rev", bins=30, log_scale=(False, True))
    return


@app.cell
def _(mo):
    mo.md(r"""
    #General Observations
    """)
    return


@app.cell
def _(TableReport, train_data):
    report = TableReport(
        train_data,
    )
    return (report,)


@app.cell
def _(report):
    report
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Among user installation features:
    - Low cardinality cols => platform, is_optin, game_type (1 can be dropped), app_id, country, campaing_type, mobile classification
    - Higher cardinality cols => ad_networkd_id, install_date, campaign_id, model, manufacturer, city

    Among game engagement features:
    - Low cardinality cols => iap_ads_count (never more than 4 ads)
    - Higher cardinality cols => rest of game engagement features
    """)
    return


@app.cell
def _(TableReport, d0_columns, train_data, user_install_features):
    TableReport(
        train_data.select(d0_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d30_columns, train_data, user_install_features):
    TableReport(
        train_data.select(d30_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d120_columns, train_data, user_install_features):
    TableReport(
        train_data.select(d120_columns+user_install_features),
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More observations


    Columns correlated
    - `platform` and `app_id` store the same info, keep only one
    - `manufacturer` also stores the `platform` info but allow more granularity for android users
    - in app revenue is pretty close to coins revenue (same for "counts")

    Miscellaneous
    - we are in a long tail distribution settings, most user generate close to no revenue, some a little and 5 to 10% of users generate most of the revenue (among them are the "whales" who are the top paying customers). For a machine learning problem that mean we should perform correctly on all those different cohorts that might have very different profiles and sizes.
    - users are mostly based in the US then Japan, BG and DE (65%,14%,11%,8%)
    - iphones are more common than androids (60/40) and naturally dominates the `model` column. After apple, samsung account for 25%
    - `city` column missing a lot of values, then tokyo and probably many US cities for less than 1% of the values each. Probably not a very discriminative column since very high cardinality. Could be dropped, further analysis needed.
    - similar conclusion for `campaign_id` column. Could be dropped, further analysis needed.
    - mobile classification dominated by tier 2. Difficult to interpret higher tier means more expensive phone most likely (tier 1 > tier 2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Segmentation analysis
    """)
    return


@app.cell
def _(datetime):
    pipeline_run_date = datetime.strptime("2024-03-13", "%Y-%m-%d")
    pipeline_run_date
    return (pipeline_run_date,)


@app.cell
def _():
    cols_to_drop = ['app_id', 'city', 'game_type'] + [f'iap_coins_rev_d{dx}' for dx in [0, 3, 7, 14, 30, 60, 90, 120]] + [f'iap_coins_count_d{dx}' for dx in [0, 3, 7, 14, 30, 60, 90, 120]]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Segmentation by 120d revenue
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=100, step=1, value=80, label="Target Revenue %")
    return (slider,)


@app.cell
def _(mo, pl, rev_users, slider):
    def calculate_pareto(df, target_pct):
        processed = (
            df.select("d120_rev")
            .filter(pl.col("d120_rev") > 0)
            .sort("d120_rev", descending=True)
            .with_columns(
                cum_rev = pl.col("d120_rev").cum_sum(),
                total_rev = pl.col("d120_rev").sum()
            )
            .with_columns(
                cum_pct = (pl.col("cum_rev") / pl.col("total_rev")) * 100
            )
        )

        users_needed = processed.filter(pl.col("cum_pct") <= target_pct).height
        total_users = processed.height
        user_pct = (users_needed / total_users) * 100

        return users_needed, user_pct

    count, u_pct = calculate_pareto(rev_users, slider.value)


    mo.vstack([
        slider,
        mo.md(f"### Results"),
        mo.stat(
            label="Users needed", 
            value=f"{count:,}", 
            caption=f"This is {u_pct:.2f}% of the paying user base."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The top 1% account for 50%, and correctly addressing 6% of the paying user base could potentially capture 80% of the revenue.And with 20% of the paying users,  you can capture 95% of the revenue.

    It is crucial to be able to identify those users early and provide them with a premium experience to maximize their lifetime value.
    """)
    return


@app.cell
def _(pipeline_run_date, pl, timedelta, train_data):
    train_data_120d = train_data.filter(pl.col('install_date') <= pipeline_run_date - timedelta(days=120))
    return (train_data_120d,)


@app.cell
def _(train_data, train_data_120d):
    train_data.height , train_data_120d.height
    return


@app.cell
def _(pl, train_data_120d):
    no_rev_users = train_data_120d.filter(pl.col("d120_rev") == 0)
    rev_users = train_data_120d.filter(pl.col("d120_rev") > 0)
    return no_rev_users, rev_users


@app.cell
def _(pl, rev_users):
    rev_users_segmented = rev_users.with_columns(
        top_rev_1pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
    )
    return (rev_users_segmented,)


@app.cell
def _(no_rev_users, pl, rev_users_segmented):
    top_rev_users = rev_users_segmented.filter(pl.col("top_rev_1pct"))
    high_rev_users = rev_users_segmented.filter(
        (pl.col("top_rev_1pct") == False) & (pl.col("top_rev_5pct"))
    )
    mid_rev_users = rev_users_segmented.filter(
        (pl.col("top_rev_1pct") == False) & (pl.col("top_rev_5pct") == False) & (pl.col("top_rev_20pct"))
    )
    low_rev_users = rev_users_segmented.filter(pl.col("top_rev_20pct") == False)
    no_rev_users.height, low_rev_users.height, mid_rev_users.height, high_rev_users.height, top_rev_users.height
    return high_rev_users, low_rev_users, mid_rev_users, top_rev_users


@app.cell
def _(TableReport, d120_columns, top_rev_users, user_install_features):
    TableReport(
        top_rev_users.select(d120_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d120_columns, high_rev_users, user_install_features):
    TableReport(
        high_rev_users.select(d120_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d120_columns, mid_rev_users, user_install_features):
    TableReport(
        mid_rev_users.select(d120_columns+user_install_features),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Observation

    It is quite complicated to segment our user base by revenue and get homogeneous groups, only a very small sample of users will drive some very high revenue so even among top paying players (1%) there are still outliers.

    With our pareto segmentation, we can still notice a pattern with:
    - top revenue users mostly spending with in-app purchase
    - medium revenue users mostly bringing add revenue
    - the high revenue cohort being a mix of both

    iOS User are slightly over-represented among top revenue generators but not by much (75% vs 60% overall).
    Country doesn't seem to have a strong effect on revenue generation either.
    One campaign seems to drive more revenue than others, further analysis may be needed but a flag on it might be interesting.
    Campaign type C is outperforming the others on top customers.
    iPhoneUnknown and iPhone13ProMax overperform on top revenue generation.
    Apple outperforms but that was expected with other columns. Google is slightly outperforming on this segment enoug to justify to keep the column `manufacturer`.
    City and mobile_classification don't seem to be good predictors alone.
    Install date doesn't seem to have a great impact. Early adopters are a bit less represented, needs to be studied more.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Early life of top customers
    """)
    return


@app.cell
def _(TableReport, d0_columns, top_rev_users, user_install_features):
    TableReport(
        top_rev_users.select(d0_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d0_columns, high_rev_users, user_install_features):
    TableReport(
        high_rev_users.select(d0_columns+user_install_features),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Early life of low/no revenue players
    """)
    return


@app.cell
def _(TableReport, d0_columns, low_rev_users, user_install_features):
    TableReport(
        low_rev_users.select(d0_columns+user_install_features),
    )
    return


@app.cell
def _(TableReport, d0_columns, no_rev_users, user_install_features):
    TableReport(
        no_rev_users.select(d0_columns+user_install_features),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Looking for business KPIs to infer segmentation early
    """)
    return


@app.cell
def _(high_rev_users, mid_rev_users, pl, top_rev_users):
    top_rev_users.filter(pl.col('d0_rev') > 0).height ,  high_rev_users.filter(pl.col('d0_rev') > 0).height,  mid_rev_users.filter(pl.col('d0_rev') > 0).height 
    return


@app.cell
def _(high_rev_users, mid_rev_users, pl, top_rev_users):
    top_rev_users.filter(pl.col('d0_rev') > 0).height/top_rev_users.height , high_rev_users.filter(pl.col('d0_rev') > 0).height/high_rev_users.height, mid_rev_users.filter(pl.col('d0_rev') > 0).height / mid_rev_users.height
    return


@app.cell
def _(high_rev_users, mid_rev_users, pl, top_rev_users):
    top_rev_users.filter(pl.col('d0_ad_rev') > 0).height/top_rev_users.height , high_rev_users.filter(pl.col('d0_ad_rev') > 0).height/high_rev_users.height, mid_rev_users.filter(pl.col('d0_ad_rev') > 0).height / mid_rev_users.height
    return


@app.cell
def _(high_rev_users, mid_rev_users, pl, top_rev_users):
    top_rev_users.filter(pl.col('d0_iap_rev') > 0).height/top_rev_users.height , high_rev_users.filter(pl.col('d0_iap_rev') > 0).height/high_rev_users.height, mid_rev_users.filter(pl.col('d0_iap_rev') > 0).height / mid_rev_users.height
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can catch 99% of top to mid performers by looking at revenue on the first day.
    On top of that, they show significatively more engagement with higher levels, sessions,...

    Moreover, in-app purchase already inform us of 30% of our top performers and 15% of the high ones
    """)
    return


@app.cell
def _(pl, rev_users):
    rev_users.filter(pl.col('d0_rev') > 0).height , rev_users.filter(pl.col('d0_rev') > 0).height/rev_users.height
    return


@app.cell
def _(pl, rev_users):
    rev_users.filter(pl.col('d0_ad_rev') > 0).height , rev_users.filter(pl.col('d0_ad_rev') > 0).height/rev_users.height
    return


@app.cell
def _(pl, rev_users):
    rev_users.filter(pl.col('d0_iap_rev') > 0).height , rev_users.filter(pl.col('d0_iap_rev') > 0).height/rev_users.height
    return


@app.cell
def _(mo):
    mo.md("""
    98.7% of revenue users will bring revenue on the first day.

    While only 1.6% will do an in-app purchase
    """)
    return


@app.cell
def _(
    high_rev_users,
    low_rev_users,
    mid_rev_users,
    no_rev_users,
    pl,
    top_rev_users,
):
    top_rev_users.filter(pl.col('game_count_d0') > 20).height/top_rev_users.height , high_rev_users.filter(pl.col('game_count_d0') > 20).height/high_rev_users.height, mid_rev_users.filter(pl.col('game_count_d0') > 20).height / mid_rev_users.height, low_rev_users.filter(pl.col('game_count_d0') > 20).height / low_rev_users.height, no_rev_users.filter(pl.col('game_count_d0') > 20).height / no_rev_users.height
    return


@app.cell
def _(low_rev_users, pl):
    low_rev_users.filter((pl.col('d0_rev') == 0) & (pl.col('game_count_d0') <= 5)).height / low_rev_users.height
    return


@app.cell
def _(
    high_rev_users,
    low_rev_users,
    mid_rev_users,
    no_rev_users,
    pl,
    top_rev_users,
):
    top_rev_users.filter(pl.col('session_count_d0') > 3).height/top_rev_users.height , high_rev_users.filter(pl.col('session_count_d0') > 3).height/high_rev_users.height, mid_rev_users.filter(pl.col('session_count_d0') > 3).height / mid_rev_users.height, low_rev_users.filter(pl.col('session_count_d0') > 3).height / low_rev_users.height, no_rev_users.filter(pl.col('session_count_d0') > 3).height / no_rev_users.height
    return


@app.cell
def _(
    high_rev_users,
    low_rev_users,
    mid_rev_users,
    no_rev_users,
    pl,
    top_rev_users,
):
    top_rev_users.filter(pl.col('current_level_d0') > 15).height/top_rev_users.height , high_rev_users.filter(pl.col('current_level_d0') > 15).height/high_rev_users.height, mid_rev_users.filter(pl.col('current_level_d0') > 15).height / mid_rev_users.height, low_rev_users.filter(pl.col('current_level_d0') > 15).height / low_rev_users.height, no_rev_users.filter(pl.col('current_level_d0') > 15).height / no_rev_users.height
    return


@app.cell
def _(pl, train_data):
    train_data.filter((pl.col('d0_rev') == 0)).height / train_data.height
    return


@app.cell
def _(pl, train_data):
    train_data.filter((pl.col('d0_rev') == 0) & (pl.col('game_count_d0') <= 20)).height / train_data.height
    return


@app.cell
def _(pl, train_data):
    train_data.filter((pl.col('d0_rev') == 0) & (pl.col('current_level_d0') <= 10)).height / train_data.height
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Conclusions:

    With D0 information it seems we can infer for a lot of users that they won't generate revenue by looking at revenue they generated and simple engagement metrics like level and game count.

    Also, the type of revenue (add vs in-app) and the D0 progression appears to give great insights in the potential of revenue generation of users.

    It would be interesting to try to first apply a classifier to identify potential revenue generators and then train dedicated regressor to predict the amount of revenue they will bring. This will reduce the amount of "outliers" since each class will be more "coherent" hopefully making the problem easier to solve.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
