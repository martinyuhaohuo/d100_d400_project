import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path

# note that functions in this module is generated through LLM (ChatGPT 5.1)

# they are modified manually to adjust style & fit the data format

# their result is also verified manually


def plot_numeric_hist(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    unit: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a histogram of a numerical column in a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name of the numerical variable.
    bins : int, optional
        Number of histogram bins (default 30).
    unit : str, optional
        the unit for x axis
    title : str, optional
        Custom plot title. If None, a default title is used.
    save_path : Path, optional
        Path to save the plot
        If None, the plot is not saved.

    Returns
    -------
    None
        Displays the histogram and optionally saves it.
    """

    # Select the data
    series = df[col].dropna()

    # Plot
    plt.figure(figsize=(7, 4))
    plt.hist(
        series,
        bins=bins,
        color="cornflowerblue",
        edgecolor="black",
        alpha=0.75,
    )

    if unit:
        plt.xlabel(col + " " + unit)
    else:
        plt.xlabel(col)
    plt.ylabel("Counts")

    if title is None:
        title = f"Histogram of {col}"
    plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        file_name = f"hist_{col}.png"
        final_path = save_path / file_name
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_hist_by_twocategory(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    bins: int = 30,
    unit: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot overlapped histograms (in percent) for two categories of a categorical variable

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the numerical and categorical variables.
    numeric_col : str
        The name of the numerical column whose distribution will be plotted.
        Values will be converted into percentage weights for the histogram.
    category_col : str
        The name of the categorical column used to split the data. Must contain
        exactly two unique non-null categories.
    bins : int, optional
        Number of bins to use in the histogram (default is 30).
    unit : str, optional
        A unit label appended to the x-axis label. If None, only the column
        name is shown.
    title : str, optional
        A custom title for the plot. If None, a default title is generated.
    save_path : pathlib.Path, optional
        If provided, the figure will be saved to this directory using a file

    Returns
    -------
    None
        The function produces a matplotlib figure directly and optionally saves
        it to designated path
    """

    # Drop NA category
    df2 = df.dropna(subset=[category_col]).copy()

    # Category values (must be exactly 2)
    categories = df2[category_col].unique()
    if len(categories) != 2:
        raise ValueError(
            "category_col must contain EXACTLY TWO non-NA categories."
        )

    cat1, cat2 = categories

    # cat proportion
    cat1_p = (
        df2[df2[category_col] == cat1][category_col].count() / df2.shape[0]
    )
    cat1_p = round(cat1_p * 100, 3)
    cat2_p = (
        df2[df2[category_col] == cat2][category_col].count() / df2.shape[0]
    )
    cat2_p = round(cat2_p * 100, 3)

    # Extract the two series
    s1 = df2.loc[df2[category_col] == cat1, numeric_col].dropna()
    s2 = df2.loc[df2[category_col] == cat2, numeric_col].dropna()

    # Convert hist from count → percentage using weights
    weights1 = [1 / len(s1)] * len(s1) if len(s1) > 0 else None
    weights2 = [1 / len(s2)] * len(s2) if len(s2) > 0 else None

    x_min = df2[numeric_col].min()
    x_max = df2[numeric_col].max()

    # Plot
    plt.figure(figsize=(7, 4))

    plt.hist(
        s1,
        bins=bins,
        weights=weights1,
        range=(x_min, x_max),
        color="steelblue",
        alpha=0.5,
        edgecolor="black",
        label=f"{category_col} = {cat1}, {cat1_p}%",
    )

    plt.hist(
        s2,
        bins=bins,
        weights=weights2,
        range=(x_min, x_max),
        color="darkorange",
        alpha=0.5,
        edgecolor="black",
        label=f"{category_col} = {cat2}, {cat2_p}%",
    )

    mean1 = s1.mean()
    plt.axvline(
        mean1,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"{cat1} mean = {mean1:.2f}",
    )

    mean2 = s2.mean()
    plt.axvline(
        mean2,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"{cat2} mean = {mean2:.2f}",
    )

    # Labels
    if unit:
        plt.xlabel(f"{numeric_col} ({unit})")
    else:
        plt.xlabel(numeric_col)

    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Title
    if title:
        plt.title(title)
    else:
        plt.title(f"Distribution of {numeric_col} by {category_col}")

    plt.tight_layout()

    # Save
    if save_path is not None:
        file_name = f"overlapped_hist_{numeric_col}_by_{category_col}.png"
        final_path = save_path / file_name
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_two_histograms(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    xlabel: str,
    bins: int = 30,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot two histograms on the same figure for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the two numerical columns.
    col1 : str
        The name of the first column.
    col2 : str
        The name of the second column.
    xlabel : str
        The xlabel of the plot
    bins : int, optional
        Number of bins for the histogram. Default is 30.
    title : str, optional
        Title of the plot.
    save_path : Path, optional
        If provided, saves the plot to this file path.

    Returns
    -------
    None
    """

    series1 = df[col1].dropna()
    series2 = df[col2].dropna()

    x_min_1 = series1.min()
    x_max_1 = series1.max()
    x_min_2 = series2.min()
    x_max_2 = series2.max()

    if x_min_1 < x_min_2:
        x_min = x_min_1
    else:
        x_min = x_min_2

    if x_max_1 > x_max_2:
        x_max = x_max_1
    else:
        x_max = x_max_2

    plt.figure(figsize=(8, 5))

    plt.hist(
        series1,
        bins=bins,
        alpha=0.5,
        label=col1,
        range=(x_min, x_max),
        density=True,
        edgecolor="black",
        color="darkorange",
    )
    plt.hist(
        series2,
        bins=bins,
        alpha=0.5,
        label=col2,
        range=(x_min, x_max),
        density=True,
        edgecolor="black",
        color="steelblue",
    )

    plt.legend()
    plt.ylabel("Density")
    plt.xlabel(xlabel)

    if title:
        plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path is not None:
        file_name = f"hist_{col1}_and_{col2}.png"
        final_path = save_path / file_name
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_horizontal_bar_counts(
    df: pd.DataFrame,
    column_cat: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a horizontal bar chart of category counts for a given categorical column.
    pd.NA values are treated as a category.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the categorical column.
    column_cat : str
        The name of the categorical column to plot.
    title : str, optional
        Title of the plot.
    save_path : Path, optional
        If given, saves the figure to this file path.

    Returns
    -------
    None
    """

    # Keep NA as a category
    series = df[column_cat].astype("object").fillna("NA")

    # Count occurrences and sort descending
    counts = series.value_counts().sort_values(
        ascending=True
    )  # ascending for horizontal bar

    # Create plot
    plt.figure(figsize=(10, max(4, len(counts) * 0.4)))
    counts.plot(
        kind="barh", color="cornflowerblue", edgecolor="black", alpha=0.75
    )

    plt.xlabel("Count")
    plt.ylabel(column_cat)
    if title:
        plt.title(title)

    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        file_name = f"horizontal_hist_{column_cat}.png"
        final_path = save_path / file_name
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_dummy_percentages(
    df: pd.DataFrame,
    dummy_cols: List[str],
    threshold: float = 0.05,  # 5% default
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot percentage of 1s for multiple dummy columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing dummy variables.
    dummy_cols : list of str
        Names of the dummy columns to plot.
    threshold : float, optional
        Horizontal line threshold (e.g., 0.05 for 5%).
    title : str, optional
        Plot title.
    save_path : Path, optional
        File path to save the plot.

    Returns
    -------
    None
    """

    # Compute percentage of 1's for each dummy column
    pct = df[dummy_cols].mean().astype(float)  # mean of 0/1 = proportion

    pct = pct.sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    plt.plot(pct.index, pct.values, linestyle="-", color="steelblue")

    # Horizontal threshold line
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {threshold*100:.1f}%",
    )

    xticks = list(range(len(pct)))
    xlabels = pct.index.tolist()

    plt.xticks(
        ticks=[i for i in xticks if i % 50 == 0],
        labels=[xlabels[i] for i in range(len(xlabels)) if i % 50 == 0],
        rotation=90,
    )

    plt.ylabel("Percentage of Counts")
    plt.xlabel("Categories")

    if title:
        plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        file_name = "category_count.png"
        final_path = save_path / file_name
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

    plt.show()


def boxplot_avg_rating_by_group(
    df: pd.DataFrame,
    rating_col: str,
    cat_col: str,
    include_na_group: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a boxplot of the rating column grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rating and categorical column.
    rating_col : str
        Name of the numerical rating column (e.g., 'avg_rating').
    cat_col : str
        Name of the categorical grouping variable.
    include_na_group : bool, optional
        If True, treat NA in cat_col as its own category ('NA').
    title : str, optional
        Title for the plot.
    save_path : Path, optional
        File path to save the figure.

    Returns
    -------
    None
    """

    df2 = df.copy()

    # Handle NA categories
    if include_na_group:
        df2[cat_col] = df2[cat_col].astype("object").fillna("NA")

    medians = df2.groupby(cat_col)[rating_col].median().sort_values()
    ordered_cats = medians.index.tolist()

    # Convert to ordered category so pandas draws boxes in correct order
    df2[cat_col] = pd.Categorical(
        df2[cat_col], categories=ordered_cats, ordered=True
    )

    # Plot
    plt.figure(figsize=(7, max(4, len(ordered_cats) * 0.4)))

    df2.boxplot(
        column=rating_col,
        by=cat_col,
        grid=False,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
        medianprops=dict(color="darkorange", linewidth=2, alpha=0.7),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="grey",
            alpha=0.5,
        ),
    )

    plt.xlabel(cat_col)
    plt.ylabel(rating_col)

    if title:
        plt.title(title)
    else:
        plt.title(f"{rating_col} by {cat_col}")

    plt.suptitle("")  # remove pandas default title
    plt.tight_layout()

    if save_path is not None:
        file_name = f"boxplot_{rating_col}_by_{cat_col}.png"
        plt.savefig(save_path / file_name, dpi=300, bbox_inches="tight")

    plt.show()


def scatter_plot_with_reg(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    unit: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create a scatter plot between two variables with a fitted regression line.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the variables.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    unit: str
        The unit of the x variabe
    title : str, optional
        Custom title for the plot.
    save_path : Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    """

    # Drop NA values first (required for regression)
    df2 = df[[x_col, y_col]].dropna()

    x = df2[x_col].values
    y = df2[y_col].values

    plt.figure(figsize=(7, 4))

    # Transparent points
    plt.scatter(
        x,
        y,
        facecolors="none",  # transparent
        edgecolors="steelblue",  # outline color
        alpha=0.5,
    )

    # Fit regression line: y = a*x + b
    a, b = np.polyfit(x, y, 1)
    reg_x = np.linspace(x.min(), x.max(), 200)
    reg_y = a * reg_x + b

    # Plot regression line in red
    plt.plot(
        reg_x,
        reg_y,
        color="red",
        linewidth=1,
        linestyle="--",
        label=f"{round(a, 3)} * {x_col} + {round(b, 3)}",
    )

    # Labels and title
    plt.xlabel(x_col + f" ({unit})")
    plt.ylabel(y_col)

    if title:
        plt.title(title)
    else:
        plt.title(f"{y_col} vs {x_col} with Regression Line")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot if a path is provided
    if save_path is not None:
        filename = f"scatter_reg_{x_col}_vs_{y_col}.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")

    plt.show()


def scatter_plot_with_quad_reg(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    unit: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create a scatter plot between two variables with a quadratic regression line.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the variables.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    unit: str
        The unit of the x variabe
    title : str, optional
        Custom title for the plot.
    save_path : Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    """

    # Drop NA for regression computation
    df2 = df[[x_col, y_col]].dropna()

    x = df2[x_col].values
    y = df2[y_col].values

    plt.figure(figsize=(7, 4))

    # Transparent scatter points
    plt.scatter(x, y, facecolors="none", edgecolors="steelblue", alpha=0.5)

    # Quadratic regression: y = a + b x + c x^2
    coeffs = np.polyfit(x, y, deg=2)  # <--- DEGREE 2
    a, b, c = coeffs

    # Generate smooth x values for plotting the curve
    reg_x = np.linspace(x.min(), x.max(), 300)
    reg_y = a * reg_x**2 + b * reg_x + c

    # Plot quadratic regression line
    plt.plot(
        reg_x,
        reg_y,
        color="red",
        linewidth=1,
        linestyle="--",
        label=f"{round(a, 3)} * {x_col}^2 + {round(b, 3)} * {x_col} + {round(c, 3)}",
    )

    # Labels and title
    plt.xlabel(x_col + f" ({unit})")
    plt.ylabel(y_col)

    if title:
        plt.title(title)
    else:
        plt.title(f"{y_col} vs {x_col} (Quadratic Regression)")

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Optional save
    if save_path is not None:
        filename = f"scatter_quad_reg_{x_col}_vs_{y_col}.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")

    plt.show()
