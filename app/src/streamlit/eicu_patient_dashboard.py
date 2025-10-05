from __future__ import annotations

import os
import io
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend no interactivo para Streamlit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns

import streamlit as st
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.cluster import KMeans

from scipy.sparse import load_npz
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="ICU Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Router de páginas ---
if "_page" not in st.session_state:
    st.session_state["_page"] = "dashboard"
# Estado inicial del selector de gráfico (evita lag)
if "viz" not in st.session_state:
    st.session_state["viz"] = "Resumen 3x3"


# Paleta / estilo base matplot
plt.style.use("default")

# -----------------------------
# Utilidades de datos
# -----------------------------
@st.cache_data(show_spinner=False)
def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Carga patient.csv y hospital.csv desde un directorio.
    Si no existe hospital.csv, devuelve sólo patient.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    p_path = os.path.join(data_dir, "patient.csv")
    h_path = os.path.join(data_dir, "hospital.csv")

    if os.path.exists(p_path):
        dfs["patient"] = read_csv_safe(p_path)
    if os.path.exists(h_path):
        dfs["hospital"] = read_csv_safe(h_path)
    return dfs

# Utilidad: carga matriz dispersa .npz (scipy)
@st.cache_data(show_spinner=False)
def load_sparse_matrix(path: str):
    return load_npz(path)

    if os.path.exists(p_path):
        dfs["patient"] = read_csv_safe(p_path)
    if os.path.exists(h_path):
        dfs["hospital"] = read_csv_safe(h_path)
    return dfs

# ----------------
# Utils 
# ----------------

def numeric_age(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    older_mask = s.str.contains(">")
    out = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    out[older_mask] = np.where(out[older_mask].notna(), out[older_mask], 90)
    return out


def preprocess_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        df["age_years"] = numeric_age(df["age"])
    else:
        df["age_years"] = np.nan

    if "unitdischargeoffset" in df.columns:
        df["icu_los_days"] = pd.to_numeric(df["unitdischargeoffset"], errors="coerce") / (60 * 24)
    else:
        df["icu_los_days"] = np.nan

    df = df[(df["icu_los_days"].between(0, 120)) | df["icu_los_days"].isna()]
    df = df[(df["age_years"].between(0, 110)) | df["age_years"].isna()]

    for c in ["gender", "ethnicity", "hospitaladmitsource", "apacheadmissiondx"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.title()
    for c in ["patientunitstayid", "hospitalid"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64").astype("string")
    return df


def top_n_index(s: pd.Series, n: int) -> pd.Index:
    return s.value_counts(dropna=False).head(n).index

# -----------------------------
# Plots
# -----------------------------

def plot_hist_los(df: pd.DataFrame, bins: int = 50, logy: bool = False, ax=None, line: bool = False):
    if "icu_los_days" not in df.columns:
        raise ValueError("Falta la columna 'icu_los_days'.")

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    x = pd.to_numeric(df["icu_los_days"], errors="coerce").dropna()

    density = bool(line)
    ax.hist(x, bins=bins, edgecolor="black", alpha=0.4, density=density)

    if line:
        sns.kdeplot(x=x, ax=ax, linewidth=2, color="blue", bw_method="scott", cut=0)

    ax.set_title("Histograma de estancia en UCI (días)")
    ax.set_xlabel("Estancia UCI (días)")
    ax.set_ylabel("Densidad" if density else "Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.4)
    if logy:
        ax.set_yscale("log")

    if created:
        fig.tight_layout()
    return fig, ax


def plot_bar_admit_source(
    df: pd.DataFrame,
    top_n: int = 8,
    hue_col: str | None = "hospitaladmitsource",
    ax=None,
    annotate: bool = True,
):
    if "hospitaladmitsource" not in df.columns:
        raise ValueError("Falta la columna 'hospitaladmitsource'.")

    data = df.copy()
    data["hospitaladmitsource"] = data["hospitaladmitsource"].fillna("Desconocido")
    if hue_col is not None:
        if hue_col not in data.columns:
            raise ValueError(f"Falta la columna de hue '{hue_col}'.")
        data[hue_col] = data[hue_col].fillna("Desconocido")

    top_categories = (
        data["hospitaladmitsource"].value_counts().head(top_n).index.tolist()
    )
    data = data[data["hospitaladmitsource"].isin(top_categories)]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        created = True
    else:
        fig = ax.figure

    if hue_col == "hospitaladmitsource":
        plot_df = (
            data["hospitaladmitsource"]
            .value_counts()
            .loc[top_categories]
            .rename_axis("hospitaladmitsource")
            .reset_index(name="count")
        )
        sns.barplot(
            data=plot_df,
            x="hospitaladmitsource",
            y="count",
            hue="hospitaladmitsource",
            order=top_categories,
            dodge=False,  
            ax=ax,
        )
    elif hue_col is not None:
        plot_df = (
            data.groupby(["hospitaladmitsource", hue_col])
            .size()
            .reset_index(name="count")
        )
        sns.barplot(
            data=plot_df,
            x="hospitaladmitsource",
            y="count",
            hue=hue_col,
            order=top_categories,
            ax=ax,
        )
    else:
        plot_df = (
            data["hospitaladmitsource"]
            .value_counts()
            .loc[top_categories]
            .rename_axis("hospitaladmitsource")
            .reset_index(name="count")
        )
        sns.barplot(
            data=plot_df,
            x="hospitaladmitsource",
            y="count",
            order=top_categories,
            ax=ax,
        )

    ax.set_title(f"Fuente de admisión hospitalaria (top {top_n})")
    ax.set_xlabel("Fuente de admisión")
    ax.set_ylabel("Pacientes")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    if annotate:
        try:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f", padding=2)
        except Exception:
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(
                    f"{int(h)}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 2),
                    textcoords="offset points",
                )

    if created:
        fig.tight_layout()
    return fig, ax


def plot_scatter_age_los(
    df: pd.DataFrame,
    n_clusters: int = 3,
    max_points: int = 50000,
    ax=None,
    line: str = "median",  # "median" | "mean" | None
):
    """
    Scatter de edad (x) vs estancia UCI (y) con:
      - Clustering KMeans para colorear puntos.
      - Línea vertical en la mediana (o media) de la edad.
    Sin regresiones ni suavizados.

    Requisitos mínimos: columnas 'age_years' y 'icu_los_days'.
    """
    if not {"age_years", "icu_los_days"} <= set(df.columns):
        raise ValueError("Faltan columnas: se requieren 'age_years' y 'icu_los_days'.")

    tmp = df[["age_years", "icu_los_days"]].dropna()
    if len(tmp) == 0:
        raise ValueError("No hay datos válidos tras eliminar NaN.")

    if len(tmp) > max_points:
        tmp = tmp.sample(max_points, random_state=42)

    n_clusters = max(1, min(n_clusters, len(tmp)))

    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(tmp[["age_years", "icu_los_days"]].to_numpy())
    except Exception:
        labels = pd.qcut(
            tmp["age_years"].rank(method="first"),
            q=n_clusters,
            labels=False,
            duplicates="drop",
        ).to_numpy()

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    for k in range(len(np.unique(labels))):
        sub = tmp[labels == k]
        ax.scatter(
            sub["age_years"],
            sub["icu_los_days"],
            s=16,
            alpha=0.5,
            label=f"Cluster {k+1}",
            zorder=1,
        )

    if line in ("median", "mean"):
        xval = tmp["age_years"].median() if line == "median" else tmp["age_years"].mean()
        ax.axvline(xval, linestyle="--", linewidth=2, label=f"{'Mediana' if line=='median' else 'Media'} edad", zorder=2)

    ax.set_title("Edad vs estancia en UCI (clusters)")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    if created:
        fig.tight_layout()
    return fig, ax


def plot_violin_by_gender(df, ax=None, order=None):
    """
    Dibuja un violinplot de estancia en UCI por género, coloreando por 'gender'
    con una paleta definida internamente.
    """
    if "gender" not in df.columns or "icu_los_days" not in df.columns:
        raise ValueError("Faltan columnas requeridas: 'gender' y 'icu_los_days'.")

    if order is None:
        order = pd.Index(df["gender"].dropna().unique())

    colors = sns.color_palette("husl", n_colors=len(order))
    palette = dict(zip(order, colors))

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    sns.violinplot(
        data=df,
        x="gender",
        y="icu_los_days",
        hue="gender",          
        order=order,
        palette=palette,       
        ax=ax
    )

    ax.set_title("Estancia en UCI por género")
    ax.set_xlabel("Género")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, linestyle="--", alpha=0.4)

    if created:
        fig.tight_layout()
    return fig, ax


def plot_bar_mean_by_dx(df: pd.DataFrame, top_n: int = 8, ax=None):
    if "apacheadmissiondx" not in df.columns:
        raise ValueError("Falta la columna 'apacheadmissiondx'.")
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    top = top_n_index(df["apacheadmissiondx"], n=top_n)
    tmp = df[df["apacheadmissiondx"].isin(top)].copy()
    med = (tmp.groupby("apacheadmissiondx")["icu_los_days"].mean().sort_values(ascending=True))
    ax.barh(med.index, med.values, edgecolor="black")
    ax.set_title(f"Estancia media por diagnóstico (top {top_n})")
    ax.set_xlabel("Estancia media (días)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_bar_mean_by_hospital(df: pd.DataFrame, top_n: int = 8, ax=None):
    if "hospitalid" not in df.columns:
        raise ValueError("Falta la columna 'hospitalid'.")
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    top = top_n_index(df["hospitalid"], n=top_n)
    tmp = df[df["hospitalid"].isin(top)].copy()
    med = (tmp.groupby("hospitalid")["icu_los_days"].mean().sort_values(ascending=False))
    ax.bar(med.index.astype(str), med.values, edgecolor="black")
    ax.set_title(f"Estancia media por hospital (top {top_n})")
    ax.set_xlabel("Hospital ID")
    ax.set_ylabel("Estancia media (días)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if created:
        fig.tight_layout()
    return fig, ax

def plot_box_los_by_ethnicity(
    df: pd.DataFrame,
    top_n: int = 5,
    min_count: int = 20,
    ax=None,
    cmap: str = "tab20",
    colors: Optional[Dict[str, str]] = None,
    showfliers: bool = True,
    annotate_median: bool = False,
):
    """
    Boxplot de estancia en UCI (icu_los_days) por etnia (ethnicity).
    - Filtra a las 'top_n' etnias más frecuentes y descarta grupos con < min_count.
    - Ordena las categorías por mediana de estancia (descendente).
    - Colorea por categoría y añade leyenda consistente.
    """
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    tmp = df[["ethnicity", "icu_los_days"]].copy()
    tmp["ethnicity"] = tmp["ethnicity"].fillna("Desconocido").astype(str).str.title()
    tmp["icu_los_days"] = pd.to_numeric(tmp["icu_los_days"], errors="coerce")
    tmp = tmp.dropna(subset=["icu_los_days"])

    top_eth = tmp["ethnicity"].value_counts().head(top_n).index
    tmp = tmp[tmp["ethnicity"].isin(top_eth)]
    counts = tmp["ethnicity"].value_counts()
    valid_eth = counts[counts >= min_count].index
    tmp = tmp[tmp["ethnicity"].isin(valid_eth)]

    if tmp.empty:
        ax.text(0.5, 0.5, "Sin datos suficientes tras el filtrado",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    order = (
        tmp.groupby("ethnicity")["icu_los_days"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        palette = [cmap_obj(i % cmap_obj.N) for i in range(len(order))]
        color_map = {e: palette[i] for i, e in enumerate(order)}
    else:
        cmap_obj = plt.get_cmap(cmap)
        fallback = [cmap_obj(i % cmap_obj.N) for i in range(len(order))]
        color_map = {e: colors.get(e, fallback[i]) for i, e in enumerate(order)}

    sns.boxplot(
        data=tmp,
        x="ethnicity",
        y="icu_los_days",
        order=order,
        palette=[color_map[e] for e in order],
        ax=ax,
        showfliers=showfliers,
        linewidth=1.5,
    )

    if annotate_median:
        medians = tmp.groupby("ethnicity")["icu_los_days"].median()
        xs = range(len(order))
        ys = [medians[e] for e in order]
        ax.scatter(
            xs, ys,
            zorder=3,
            s=40,
            c=[color_map[e] for e in order],
            edgecolor="black",
            linewidths=0.6
        )

    ax.set_title("Estancia UCI por etnia (boxplot)")
    ax.set_xlabel("Etnia")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(order, rotation=25, ha="right")

    handles = [mpatches.Patch(facecolor=color_map[e], edgecolor="black", label=e) for e in order]
    ax.legend(handles=handles, title="Etnia", bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0)

    if created:
        fig.tight_layout()
    return fig, ax


def plot_height_hist(
    df: pd.DataFrame,
    ax=None,
    bins: int = 40,
    density: bool = False,
    clip: tuple | None = (120, 210),
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    # Altura -> cm
    h = pd.to_numeric(df.get("admissionheight"), errors="coerce")
    height_cm = np.where(h > 3, h, h * 100.0)
    height_cm = pd.Series(height_cm, index=df.index).dropna()

    if clip is not None:
        lo, hi = clip
        height_cm = height_cm[(height_cm >= lo) & (height_cm <= hi)]

    if height_cm.empty:
        ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center", transform=ax.transAxes)
        if created:
            fig.tight_layout()
        return fig, ax

    counts, edges = np.histogram(height_cm.values, bins=bins, density=density)
    bin_widths = np.diff(edges)
    bin_width = float(bin_widths.mean())

    tmp = pd.DataFrame({"height_cm": height_cm})
    bins_groups = [-np.inf, 170, 180, np.inf]
    labels = ["<170", "170–180", "≥180"]
    tmp["height_group"] = pd.cut(tmp["height_cm"], bins=bins_groups, labels=labels, right=False)

    palette = {"<170": "#1f77b4", "170–180": "#ff7f0e", "≥180": "#2ca02c"}

    multiple = "stack" if not density else "layer"
    stat = "density" if density else "count"
    sns.histplot(
        data=tmp,
        x="height_cm",
        hue="height_group",
        bins=edges,
        multiple=multiple,
        stat=stat,
        edgecolor="black",
        alpha=0.8,
        palette=palette,
        ax=ax,
        legend=False,
    )

    sns.kdeplot(height_cm.values, ax=ax, bw_method="scott", cut=0, linewidth=2)

    if not density:
        line = ax.lines[-1]
        y = line.get_ydata()
        y_counts = y * len(height_cm) * bin_width
        line.set_ydata(y_counts)

    present = [g for g in labels if g in tmp["height_group"].dropna().unique().tolist()]
    if present:
        handles = [Patch(facecolor=palette[g], edgecolor="black", label=g) for g in present]
        ax.legend(handles=handles, title="Grupo de altura (cm)", loc="best")

    ax.set_title("Distribución de altura de admisión (cm)")
    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Densidad" if density else "Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.35)

    if created:
        fig.tight_layout()
    return fig, ax


def plot_pie_hospital_admit_source(
    df: pd.DataFrame,
    top_n: int = 5,
    ax=None,
    show_counts: bool = True,
    label_radius: float = 1.28,
    line_radius: float = 1.02,
    min_sep: float = 0.125,
    clip_low: float = -0.125,
    clip_high: float = 0.90,
    title_y: float = 1.06,
    top_adjust: float = 0.88,
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created = True
    else:
        fig = ax.figure

    s = df["hospitaladmitsource"].fillna("Desconocido").astype(str).str.title()
    counts = s.value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        otros = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Otros": otros})])

    labels = counts.index.to_list()
    vals = counts.values
    total = vals.sum()
    pcts = (100.0 * vals / total) if total > 0 else np.zeros_like(vals, dtype=float)

    wedges, _ = ax.pie(vals, startangle=90, counterclock=False, labels=None, autopct=None)
    ax.axis("equal")

    def _spread_on_side(items, min_sep, low, high):
        if not items:
            return
        items.sort(key=lambda d: d["y"])  # ordenar por y original
        items[0]["y_adj"] = float(np.clip(items[0]["y"], low, high))
        for i in range(1, len(items)):
            yi = max(items[i]["y"], items[i - 1]["y_adj"] + min_sep)
            items[i]["y_adj"] = yi
        over = items[-1]["y_adj"] - high
        if over > 0:
            items[-1]["y_adj"] = high
            for i in range(len(items) - 2, -1, -1):
                yi = min(items[i]["y_adj"], items[i + 1]["y_adj"] - min_sep)
                items[i]["y_adj"] = yi
        for it in items:
            it["y_adj"] = float(np.clip(it["y_adj"], low, high))

    left_items, right_items = [], []
    for wedge, lab, pct, n in zip(wedges, labels, pcts, vals):
        theta = 0.5 * (wedge.theta1 + wedge.theta2)
        rad = np.deg2rad(theta)
        x = float(np.cos(rad))
        y = float(np.sin(rad))
        item = {"theta": theta, "x": x, "y": y, "lab": lab, "pct": pct, "n": int(n)}
        (right_items if x >= 0 else left_items).append(item)

    _spread_on_side(right_items, min_sep=min_sep, low=clip_low, high=clip_high)
    _spread_on_side(left_items, min_sep=min_sep, low=clip_low, high=clip_high)

    for items, side in ((right_items, "right"), (left_items, "left")):
        for it in items:
            x, y, y_adj = it["x"], it["y"], it["y_adj"]
            xy = (line_radius * x, line_radius * y)
            tx = label_radius * (1 if side == "right" else -1)
            ty = label_radius * y_adj
            ha = "left" if side == "right" else "right"

            txt = f"{it['lab']}: {it['pct']:.1f}%"
            if show_counts:
                txt += f" (n={it['n']})"

            ax.annotate(
                txt,
                xy=xy,
                xytext=(tx, ty),
                ha=ha,
                va="center",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=1.2,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle=f"angle3,angleA=0,angleB={it['theta']}",
                ),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
                zorder=1,
            )

    ax.set_title(f"Máxima ocupación ({top_n})", y=title_y)
    if created:
        fig.subplots_adjust(top=top_adjust)
    return fig, ax


def plot_dashboard_3x3(df, df_h):
    fig, axs = plt.subplots(3, 3, figsize=(25, 18))
    axs = np.asarray(axs)

    plot_hist_los(df, line=density_hist, bins=bins_hist, ax=axs[0, 0])
    plot_violin_by_gender(df, ax=axs[0, 1])
    plot_bar_admit_source(df, top_n=top_n, ax=axs[0, 2])

    plot_scatter_age_los(df, n_clusters=cluster_n, line=trendline, ax=axs[1, 0])
    plot_bar_mean_by_dx(df, top_n=top_n_diag, ax=axs[1, 1])
    plot_patients_by_region(df_h, top_n=top_n_hosp, ax=axs[1, 2])


    plot_box_los_by_ethnicity(df, top_n=5, min_count=20, ax=axs[2, 0])
    plot_height_hist(df, bins=40, density=density_height, clip=(120, 210), ax=axs[2, 1])
    plot_pie_hospital_admit_source(df, top_n=top_n_zone, ax=axs[2, 2])

    fig.tight_layout()
    return fig, axs

# -----------------------------
# Pacientes por región (hospital.csv)
# -----------------------------

def merge_patient_with_region(df_patient: pd.DataFrame, hospital_df: pd.DataFrame) -> pd.DataFrame:
    hosp = hospital_df.copy()
    hosp.columns = hosp.columns.str.lower()
    region_col = next((c for c in ["region", "hospitalregion", "región"] if c in hosp.columns), None)
    if region_col is None:
        raise ValueError(
            f"No encontré columna 'region' en dfs['hospital']. Columnas disponibles: {list(hosp.columns)}"
        )

    dfp = df_patient.copy()
    dfp["hospitalid"] = dfp["hospitalid"].astype(str)

    hosp = hosp[["hospitalid", region_col]].drop_duplicates()
    hosp["hospitalid"] = hosp["hospitalid"].astype(str)

    merged = dfp.merge(hosp, on="hospitalid", how="left")
    if region_col != "region":
        merged = merged.rename(columns={region_col: "region"})
    return merged


def plot_patients_by_region(df_patient_with_region: pd.DataFrame, top_n: int | None = None, ax=None):
    """
    Histograma de pacientes por región (categorías discretas).
    - Si se pasa top_n, se muestran solo las top N regiones más frecuentes.
    - Orden ascendente por recuento, como en la versión original.
    """
    if "region" not in df_patient_with_region.columns:
        raise ValueError("El DataFrame no tiene la columna 'region'. Ejecuta primero el merge.")

    series = df_patient_with_region["region"].fillna("Desconocido").astype(str)

    counts = series.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    counts = counts.sort_values(ascending=True)

    cats = counts.index.tolist()
    series = series[series.isin(cats)]

    cat = pd.Categorical(series, categories=cats, ordered=True)
    codes = cat.codes  

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        created = True
    else:
        fig = ax.figure

    bins = np.arange(len(cats) + 1) - 0.5
    ax.hist(codes, bins=bins, edgecolor="black")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=0)
    ax.set_title("Pacientes por región")
    ax.set_xlabel("Región")
    ax.set_ylabel("Número de pacientes")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(counts.values):
        ax.text(i, v, f" {v}", ha="center", va="bottom")

    if created:
        fig.tight_layout()
    return fig, ax

# -----------------------------
# Helpers UI
# -----------------------------

@st.cache_data(show_spinner=False)
def load_sparse_matrix(path: str):
    return load_npz(path)

@st.cache_resource(show_spinner=False)
def find_model_file(models_dir: str, name_hints: list[str], exts=(".joblib", ".pkl", ".json", ".bin", ".ubj")) -> str | None:
    if not os.path.isdir(models_dir):
        return None
    cand = []
    for root, _, files in os.walk(models_dir):
        for f in files:
            lf = f.lower()
            if not any(lf.endswith(ext) for ext in exts):
                continue
            if any(h in lf for h in name_hints):
                cand.append(os.path.join(root, f))
    # prioriza joblib/pkl sobre json/bin
    pref = sorted(cand, key=lambda p: (0 if os.path.splitext(p)[1] in (".joblib", ".pkl") else 1, len(p)))
    return pref[0] if pref else None

def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def render_plot(fig: plt.Figure, caption: str):
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "⬇️ Descargar PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"{caption}.png",
        mime="image/png",
    )

# -----------------------------
# Sidebar — controles
# -----------------------------
with st.sidebar:
    st.header("🧭 Navegación")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Dashboard", use_container_width=True):
            st.session_state["_page"] = "dashboard"
    with col_nav2:
        if st.button("Predicciones", use_container_width=True):
            st.session_state["_page"] = "predicciones"

    st.header("⚙️ Configuración de datos")
    default_dir = st.session_state.get("_data_dir", "/app/app/db/csv_clean")
    data_dir = st.text_input("Directorio con CSV", value=default_dir, help="Debe contener patient.csv y (opcional) hospital.csv")
    st.session_state["_data_dir"] = data_dir

    st.markdown("---")
    st.header("📊 Visualización")
    _viz_options = [
        "Resumen 3x3",
        "Estancia media UCI (Días)",
        "Estancia media UCI (Género)",
        "Admisión",
        "Estancia media UCI (Edad)",
        "Estancia media UCI (Diagnosticos)",
        "Estancia media UCI (Hospital)",
        "Ethnicity",
        "Height",
        "Zona UCI",
    ]
    _selected = st.radio("Elige un gráfico", _viz_options, key="viz")

    st.header("🎛️ Parámetros de gráficos")
    bins_hist = st.slider("Bins histograma estacia media", 10, 150, 50, step=5)
    density_hist = st.checkbox("Tendencia", value=False)

    top_n = st.slider("Top N categorías admisiones", 3, 13, 8)

    cluster_n = st.slider("N clusters (Edad vs Estancia UCI)", 2, 5, 3) 
    opcion = st.selectbox("Tendencia scatter", ["Ninguna", "Mediana", "Media"], index=0)
    trendline = {"Ninguna": None, "Mediana": "median", "Media": "mean"}[opcion]

    top_n_diag = st.slider("Top N categorías diagnosticos", 3, 10, 8)
    top_n_hosp = st.slider("Top N hospitales", 2, 5, 3)

    density_height = st.checkbox("Normalizar altura", value=False)

    with st.sidebar:
        top_n_zone = st.number_input(
            "Top N zonas",
            min_value=2,
            max_value=8,
            value=5,
            step=1,
            help="Número de zonas a mostrar en el pie. (max 8)"
        )

# -----------------------------
# Carga de datos
# -----------------------------

dfs = {}

dfs = load_tables(data_dir)

st.title("🩺 ICU Dashboard")
st.write(
    "Dashboard interactivo para explorar datos de pacientes en UCI basados en el conjunto de datos eICU."
)

if "patient" not in dfs:
    st.info(
        "Carga `patient.csv` desde la barra lateral o indica un directorio que lo contenga para continuar."
    )
    st.stop()

patient_raw = dfs["patient"]
df_patient = preprocess_patient(patient_raw)

# ------ Página: Predicciones ------
if st.session_state.get("_page") == "predicciones":
    st.title("🔮 Predicciones — Test set (VotingRegressor)")
    st.write(
        "Selecciona un paciente de la tabla de abajo. Con el índice de la fila sacamos la predicción "
        "de su estancia en UCI con el **model_VotingRegressor.joblib**."
    )

    colp1, colp2 = st.columns(2)
    with colp1:
        path_xtest_csv = st.text_input("Ruta a X_test.csv", value="/app/app/src/test/X_test.csv")
        path_sparse = st.text_input("Ruta a X_T_test.npz (matriz dispersa)", value="/app/app/src/test/X_T_test.npz")
        path_ytest = st.text_input("Ruta a y_test.csv (opcional)", value="/app/app/src/test/y_test.csv")
    with colp2:
        model_path = st.text_input("Ruta a model_VotingRegressor.joblib", value="/app/app/src/models/model_VotingRegressor.joblib")
        up_model = st.file_uploader("Subir model_VotingRegressor.joblib (opcional)", type=["joblib", "pkl"])

    df_xtest = None
    try:
        df_xtest = pd.read_csv(path_xtest_csv)
    except Exception as e:
        st.error(f"No pude leer X_test.csv: {e}")

    X_T_test = None
    if os.path.exists(path_sparse):
        try:
            X_T_test = load_sparse_matrix(path_sparse)
        except Exception as e:
            st.error(f"No pude leer la matriz dispersa: {e}")
    else:
        st.warning(f"No existe el archivo: {path_sparse}")

    if (df_xtest is not None) and (X_T_test is not None) and (X_T_test.shape[0] != len(df_xtest)):
        st.warning(f"Tamaño inconsistente entre X_test.csv (n={len(df_xtest)}) y X_T_test.npz (n={X_T_test.shape[0]}). Revisa el preprocesado.")

    idx = None
    if df_xtest is not None:
        st.subheader("👤 Selección de paciente")
        id_col = None
        for c in ["patientunitstayid", "unitstayid", "patientid", "row_id", "subject_id"]:
            if c in df_xtest.columns:
                id_col = c
                break
        if id_col:
            ids = df_xtest[id_col].astype(str).tolist()
            sel_id = st.selectbox("Elige un identificador", ids, index=0)
            idx = int(df_xtest.index[df_xtest[id_col].astype(str) == str(sel_id)][0])
            st.caption(f"Fila seleccionada: índice {idx}")
        else:
            idx = st.number_input("Fila (0-index)", min_value=0, max_value=max(0, len(df_xtest)-1), value=0, step=1)

        with st.expander("🔎 Vista rápida de X_test.csv"):
            st.dataframe(df_xtest.head(50), use_container_width=True)

    y_test = None
    if path_ytest and os.path.exists(path_ytest):
        try:
            y_test = pd.read_csv(path_ytest).squeeze()
        except Exception as e:
            st.warning(f"No pude leer y_test.csv: {e}")

    st.subheader("🧠 Modelo: VotingRegressor")
    model = None
    try:
        if up_model is not None:
            model = joblib.load(up_model)
        elif os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error("No se encontró `model_VotingRegressor.joblib`. Indica la ruta correcta o súbelo.")
    except Exception as e:
        st.error(f"No pude cargar el modelo VotingRegressor: {e}")

    if model is None:
        st.stop()

    colb1, colb2 = st.columns(2)
    do_pred_row = colb1.button("🎯 Predecir paciente seleccionado")
    do_pred_all = colb2.button("📈 Predecir y evaluar todo el test")

    if do_pred_row:
        if (X_T_test is None) or (df_xtest is None):
            st.error("Faltan X_T_test o X_test.csv para localizar la fila.")
        else:
            try:
                row_idx = int(idx)
                x_row = X_T_test.getrow(row_idx)
                y_hat = float(model.predict(x_row)[0])
                st.success(f"Predicción para la fila {row_idx}: **{y_hat:.3f}** (minutos)")
                if y_test is not None and len(y_test) > row_idx:
                    gt = float(y_test.iloc[row_idx]) if hasattr(y_test, 'iloc') else float(y_test[row_idx])
                    st.info(f"Ground truth fila {row_idx}: **{gt:.3f}** (minutos)")
                    st.metric("Error absoluto", value=f"{abs(gt - y_hat):.3f}")
            except Exception as e:
                st.error(f"No se pudo predecir la fila {idx}: {e}")

    if do_pred_all:
        if X_T_test is None:
            st.error("Falta X_T_test para predecir el conjunto completo.")
        else:
            with st.spinner("Calculando predicciones en el test..."):
                try:
                    pred = model.predict(X_T_test)
                    st.success(f"Predicciones generadas: {len(pred):,}".replace(",", "."))
                    if y_test is not None and len(y_test) == len(pred):
                        try:
                            mae = mean_absolute_error(y_test, pred)
                            r2 = r2_score(y_test, pred)
                            st.write(f"**MAE test:** {mae:.4f}")
                            st.write(f"**R² test:** {r2:.4f}")
                        except Exception as e:
                            st.warning(f"No pude calcular métricas: {e}")
                    elif y_test is not None:
                        st.warning(f"Dimensión inconsistente: y_test={len(y_test)} vs pred={len(pred)}. No calculo métricas.")

                    out = pd.DataFrame({"pred": pred})
                    if df_xtest is not None:
                        out = pd.concat([df_xtest.reset_index(drop=True), out], axis=1)
                    if y_test is not None and len(y_test) == len(pred):
                        out["y_test"] = np.asarray(y_test)
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Descargar predicciones (CSV)", data=csv_bytes, file_name="predicciones_test.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Fallo en predicción global: {e}")

    st.stop()

# KPI rápidos
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Pacientes", value=f"{len(df_patient):,}".replace(",", "."))
with c2:
    st.metric("Edad media", value=f"{df_patient['age_years'].mean():.1f} años")
with c3:
    st.metric("Mediana estancia en UCI", value=f"{df_patient['icu_los_days'].median():.1f} días")

# Datos (opcional)
with st.expander("🔎 Ver primeras filas de patient"):
    st.dataframe(df_patient.head(50), use_container_width=True)

# === Visor simple de gráficos (sin tabs) ===

st.subheader("📈 Visualización actual")
_selected = st.session_state.get("viz", "Resumen 3x3")

df_patient_region = merge_patient_with_region(df_patient, dfs["hospital"])

if _selected == "Resumen 3x3":
    fig, _ = plot_dashboard_3x3(df_patient, df_patient_region)
    render_plot(fig, "dashboard_3x3")
elif _selected == "Estancia media UCI (Días)":
    fig, _ = plot_hist_los(df_patient, line=density_hist, bins=bins_hist)
    render_plot(fig, "los_hist")
elif _selected == "Estancia media UCI (Género)":
    try:
        fig, _ = plot_violin_by_gender(df_patient)
        render_plot(fig, "los_box_genero")
    except Exception as e:
        st.warning(f"No se pudo generar el boxplot: {e}")
elif _selected == "Admisión":
    try:
        fig, _ = plot_bar_admit_source(df_patient, top_n=top_n)
        render_plot(fig, "admission_source")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Estancia media UCI (Edad)":
    fig, _ = plot_scatter_age_los(df_patient, n_clusters=cluster_n, line=trendline)
    render_plot(fig, "age_vs_los")
elif _selected == "Estancia media UCI (Diagnosticos)":
    try:
        fig, _ = plot_bar_mean_by_dx(df_patient, top_n=top_n_diag)
        render_plot(fig, "los_por_diagnostico")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Estancia media UCI (Hospital)":
    try:
        fig, _ = plot_patients_by_region(df_patient_region, top_n=top_n_hosp)
        render_plot(fig, "los_por_hospital")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Ethnicity":
    fig, _ = plot_box_los_by_ethnicity(df_patient, top_n=5, min_count=20)
    render_plot(fig, "los_por_etnia")
elif _selected == "Height":
    fig, _ = plot_height_hist(df_patient, bins=40, density=density_height, clip=(120, 210))
    render_plot(fig, "altura_hist_kde")
elif _selected == "Zona UCI":
    fig, _ = plot_pie_hospital_admit_source(df_patient, top_n=top_n_zone)
    render_plot(fig, "admission_pie")

st.caption("Alejandro Cortijo Benito -- 2025 (alejandro.cortijo.benito@alumnos.upm.es)")