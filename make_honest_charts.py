"""Generate the FOR chart and the full-context charts used by the CSS reveals.

Usage:
    python make_honest_charts.py path/to/allegations.csv

The input is the ProPublica CCRB allegations file. The script expects the
``year_received``, ``fado_type``, and ``board_disposition`` columns and writes
PNG files to this repository's ``images`` directory.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


YEAR_COL = "year_received"
FADO_COL = "fado_type"
OUTCOME_COL = "board_disposition"
PERSUASIVE_WINDOW = (2007, 2019)
OUT_DIR = Path(__file__).resolve().parent / "images"

OUTCOMES = ["Substantiated", "Exonerated", "Unsubstantiated"]
COLORS = {
    "Substantiated": "#b42318",
    "Exonerated": "#3973a6",
    "Unsubstantiated": "#89929b",
}


def bucket_outcome(value: object) -> str:
    """Collapse the disposition subtypes without merging distinct outcomes."""
    value = str(value).strip().lower()
    if value.startswith("substantiated"):
        return "Substantiated"
    if value.startswith("exonerated"):
        return "Exonerated"
    if value.startswith("unsubstantiated"):
        return "Unsubstantiated"
    return "Other"


def prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {YEAR_COL, FADO_COL, OUTCOME_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    df = df.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, FADO_COL, OUTCOME_COL])
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df["outcome"] = df[OUTCOME_COL].map(bucket_outcome)
    unexpected = sorted(df.loc[df["outcome"].eq("Other"), OUTCOME_COL].unique())
    if unexpected:
        raise ValueError(
            "Unrecognized board_disposition value(s): " + ", ".join(unexpected)
        )
    return df


def save_for_honest(df: pd.DataFrame) -> None:
    annual_rate = (
        df.assign(substantiated=df["outcome"].eq("Substantiated"))
        .groupby(YEAR_COL)["substantiated"]
        .mean()
        .mul(100)
        .sort_index()
    )
    # The 2020 records are incomplete because many cases had not closed yet.
    # Omitting that partial year prevents the artificial zero from dominating
    # the visual comparison with completed years.
    annual_rate = annual_rate[annual_rate.index < 2020]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(
        annual_rate.index,
        annual_rate.values,
        color="#216e65",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax.axvspan(
        *PERSUASIVE_WINDOW,
        color="#e76f51",
        alpha=0.14,
        label=(
            "Window emphasized in the persuasive chart "
            f"({PERSUASIVE_WINDOW[0]}–{PERSUASIVE_WINDOW[1]})"
        ),
    )
    ax.set_xlim(annual_rate.index.min(), annual_rate.index.max())
    ax.set_ylim(0, max(40, annual_rate.max() + 3))
    ax.set_xlabel("Year complaint was received")
    ax.set_ylabel("Allegations substantiated (%)")
    ax.set_title(
        "The selected window is part of a longer, uneven history",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", color="#d9dde1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Note: 2020 is omitted because the data for that year is incomplete due "
        "to unclosed cases.\nThe dataset includes only active-duty officers with "
        "at least one substantiated allegation.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "for-proposition-honest.png", dpi=180)
    plt.close(fig)


def save_for_persuasive(df: pd.DataFrame) -> None:
    """Create the deliberately persuasive 2007–2019 chart described in the report."""
    annual_rate = (
        df.assign(substantiated=df["outcome"].eq("Substantiated"))
        .groupby(YEAR_COL)["substantiated"]
        .mean()
        .mul(100)
        .sort_index()
        .loc[PERSUASIVE_WINDOW[0] : PERSUASIVE_WINDOW[1]]
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    color = "#d95f0e"
    ax.fill_between(
        annual_rate.index,
        10,
        annual_rate.values,
        color=color,
        alpha=0.18,
    )
    ax.plot(
        annual_rate.index,
        annual_rate.values,
        color=color,
        linewidth=3,
        marker="o",
        markersize=5,
    )

    for year in (2007, 2015):
        value = annual_rate.loc[year]
        ax.scatter(year, value, s=75, color=color, zorder=3)
        ax.annotate(
            f"{year}: {value:.0f}%",
            xy=(year, value),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            color="#8f3b00",
            fontweight="bold",
        )

    ax.set_ylim(10, 40)
    ax.set_xlim(*PERSUASIVE_WINDOW)
    ax.set_xlabel("Year complaint was received")
    ax.set_ylabel("Allegations substantiated (%)")
    ax.set_title(
        "Findings Against Officers More Than Doubled Since 2007",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax.grid(axis="y", color="#e1e4e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Source: ProPublica CCRB records. Share of allegations substantiated by "
        "year received, 2007–2019.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "for-proposition.png", dpi=180)
    plt.close(fig)


def save_against_honest(df: pd.DataFrame) -> None:
    tab = pd.crosstab(df[FADO_COL], df["outcome"], normalize="index").mul(100)
    tab = tab.reindex(columns=OUTCOMES, fill_value=0).sort_values("Substantiated")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    left = pd.Series(0.0, index=tab.index)
    for outcome in OUTCOMES:
        values = tab[outcome]
        ax.barh(
            tab.index,
            values,
            left=left,
            color=COLORS[outcome],
            label=outcome,
        )
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 4:
                ax.text(
                    start + value / 2,
                    y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
        left += values

    min_year = int(df[YEAR_COL].min())
    max_year = int(df[YEAR_COL].max())
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of allegations (%)")
    ax.set_title(
        f"Full {min_year}–{max_year} record separates each investigative outcome",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    ax.grid(axis="x", color="#e1e4e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Source: ProPublica CCRB records. The released dataset excludes allegations "
        "classified as unfounded and includes only active-duty officers with at least "
        "one substantiated allegation.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(
        OUT_DIR / "against-proposition-honest.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_against_persuasive(df: pd.DataFrame) -> None:
    """Create the deliberately persuasive 2000–2009 outcome comparison."""
    window = df[df[YEAR_COL].between(2000, 2009)].copy()
    window["result"] = window["outcome"].where(
        window["outcome"].eq("Substantiated"), "Exonerated or Unsubstantiated"
    )
    tab = pd.crosstab(window[FADO_COL], window["result"], normalize="index").mul(100)
    tab = tab.reindex(
        columns=["Substantiated", "Exonerated or Unsubstantiated"], fill_value=0
    ).sort_values("Substantiated", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = range(len(tab.index))
    height = 0.34
    ax.barh(
        [position - height / 2 for position in positions],
        tab["Substantiated"],
        height=height,
        color="#b42318",
        label="Substantiated",
    )
    ax.barh(
        [position + height / 2 for position in positions],
        tab["Exonerated or Unsubstantiated"],
        height=height,
        color="#b8bec5",
        label="Exonerated or Unsubstantiated",
    )

    for y, value in enumerate(tab["Substantiated"]):
        ax.text(value + 1, y - height / 2, f"{value:.1f}%", va="center", fontsize=9)
    for y, value in enumerate(tab["Exonerated or Unsubstantiated"]):
        ax.text(value + 1, y + height / 2, f"{value:.1f}%", va="center", fontsize=9)

    ax.set_yticks(list(positions), tab.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xlabel("Share of allegation outcomes (%)")
    ax.set_title(
        "Where Complaints Matter Most, Findings Are Rarest\n"
        "Force Allegations Expose a System Lacking Accountability",
        fontsize=13,
        fontweight="bold",
        color="#7f1d1d",
        pad=14,
    )
    ax.text(
        0.5,
        1.015,
        "Case outcomes by allegation type (FADO Type), 2000–2009",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="x", color="#e1e4e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Source: ProPublica CCRB records. Complaints received 2000–2009.",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_DIR / "against-proposition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_honest_charts.py path/to/allegations.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data(sys.argv[1])
    save_for_persuasive(data)
    save_for_honest(data)
    save_against_persuasive(data)
    save_against_honest(data)
    print(f"Wrote chart images to {OUT_DIR}")


if __name__ == "__main__":
    main()
