# Streamlit ICU Dashboard — basado en tu script de Jupyter
# ---------------------------------------------------------
# Ejecuta con:  streamlit run streamlit_dashboard_app.py
# Requisitos (mínimo): streamlit, pandas, numpy, matplotlib, seaborn

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

# === Nuevos imports para predicción ===
from scipy.sparse import load_npz
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
try:
    import xgboost as xgb
except Exception:
    xgb = None

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
# Estado inicial del selector de gráfico (evita doble clic)
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

# -----------------------------
# Preprocesado (adaptado de tu script)
# -----------------------------

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
# Funciones de plotting (adaptadas/copiadas de tu script)
# -----------------------------

def plot_hist_los(df: pd.DataFrame, bins: int = 50, logy: bool = False, ax=None):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    x = df["icu_los_days"].dropna()
    ax.hist(x, bins=bins, edgecolor="black")
    ax.set_title("Histograma de estancia en UCI (días)")
    ax.set_xlabel("Estancia UCI (días)")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, linestyle="--", alpha=0.4)
    if logy:
        ax.set_yscale("log")
    if created:
        fig.tight_layout()
    return fig, ax


def plot_box_by_gender(df: pd.DataFrame, ax=None):
    if "gender" not in df.columns:
        raise ValueError("Falta la columna 'gender'.")
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    genders = list(df["gender"].dropna().unique())
    data = [df.loc[df["gender"] == g, "icu_los_days"].dropna() for g in genders]
    ax.boxplot(data, labels=genders, showfliers=True)
    ax.set_title("Estancia en UCI por género")
    ax.set_xlabel("Género")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_bar_admit_source(df: pd.DataFrame, top_n: int = 8, ax=None):
    if "hospitaladmitsource" not in df.columns:
        raise ValueError("Falta la columna 'hospitaladmitsource'.")
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    counts = df["hospitaladmitsource"].fillna("Desconocido").value_counts().head(top_n)
    ax.bar(counts.index, counts.values, edgecolor="black")
    ax.set_title(f"Fuente de admisión hospitalaria (top {top_n})")
    ax.set_xlabel("Fuente de admisión")
    ax.set_ylabel("Pacientes")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    if created:
        fig.tight_layout()
    return fig, ax


def plot_scatter_age_los(
    df: pd.DataFrame,
    max_points: int = 50000,
    ax=None,
    trendline: bool = True,
    method: str = "linear",  # "linear" | "lowess"
    smooth_frac: float = 0.25,
    colorize_bins: bool = True,
    add_kde: bool = True,
    kde_levels: int = 10,
    kde_fill: bool = True,
    kde_alpha: float = 0.25,
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    tmp = df[["age_years", "icu_los_days"]].dropna()
    if len(tmp) > max_points:
        tmp = tmp.sample(max_points, random_state=42)

    # KDE 2D
    if add_kde and len(tmp) >= 10:
        try:
            sns.kdeplot(
                x=tmp["age_years"],
                y=tmp["icu_los_days"],
                ax=ax,
                levels=kde_levels,
                fill=kde_fill,
                alpha=kde_alpha,
                thresh=0,
                bw_method="scott",
                cmap="Greys",
                linewidths=1,
                zorder=0,
            )
        except Exception:
            pass

    # Puntos
    if colorize_bins:
        bins = [-np.inf, 30, 60, 80, np.inf]
        labels = ["<30", "30–60", "60–80", "80+"]
        tmp = tmp.copy()
        tmp["age_bin"] = pd.cut(
            tmp["age_years"], bins=bins, labels=labels, right=False, include_lowest=True
        )
        for label, sub in tmp.groupby("age_bin", sort=False):
            if len(sub) == 0:
                continue
            ax.scatter(sub["age_years"], sub["icu_los_days"], alpha=0.5, s=20, label=str(label), zorder=1)
        ax.legend(title="Edad (años)", loc="best")
    else:
        ax.scatter(tmp["age_years"], tmp["icu_los_days"], alpha=0.5, s=20, zorder=1)

    # Tendencia
    if trendline and len(tmp) >= 2:
        x = tmp["age_years"].to_numpy()
        y = tmp["icu_los_days"].to_numpy()
        if method == "lowess":
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                sm = lowess(y, x, frac=smooth_frac, it=0, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], linewidth=2, label="Tendencia (LOWESS)", zorder=2)
            except Exception:
                p = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 200)
                ax.plot(xx, np.polyval(p, xx), linewidth=2, label="Tendencia (lineal)", zorder=2)
                ax.legend(loc="best")
        elif method == "linear":
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, np.polyval(p, xx), linewidth=2, label="Tendencia (lineal)", zorder=2)
            ax.legend(loc="best")

    ax.set_title("Edad vs estancia en UCI")
    ax.set_xlabel("Edad (años)")
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


def plot_violin_los_by_ethnicity(
    df: pd.DataFrame,
    top_n: int = 5,
    min_count: int = 20,
    ax=None,
    cmap: str = "tab20",
    colors: Optional[Dict[str, str]] = None,
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    tmp = df[["ethnicity", "icu_los_days"]].copy()
    tmp["ethnicity"] = tmp["ethnicity"].fillna("Desconocido").astype(str).str.title()
    tmp = tmp.dropna(subset=["icu_los_days"])

    top_eth = tmp["ethnicity"].value_counts().head(top_n).index
    tmp = tmp[tmp["ethnicity"].isin(top_eth)]

    counts = tmp["ethnicity"].value_counts()
    valid_eth = counts[counts >= min_count].index
    tmp = tmp[tmp["ethnicity"].isin(valid_eth)]

    if tmp.empty:
        ax.text(0.5, 0.5, "Sin datos suficientes tras el filtrado", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    order = (
        tmp.groupby("ethnicity")["icu_los_days"].median().sort_values(ascending=False).index.tolist()
    )

    data = [tmp.loc[tmp["ethnicity"] == e, "icu_los_days"].values for e in order]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        palette = [cmap_obj(i % cmap_obj.N) for i in range(len(order))]
        color_map = {e: palette[i] for i, e in enumerate(order)}
    else:
        cmap_obj = plt.get_cmap(cmap)
        palette = [cmap_obj(i % cmap_obj.N) for i in range(len(order))]
        color_map = {e: colors.get(e, palette[i]) for i, e in enumerate(order)}

    for pc, e in zip(parts["bodies"], order):
        pc.set_facecolor(color_map[e])
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)

    med = [np.median(d) if len(d) else np.nan for d in data]
    ax.scatter(
        range(1, len(data) + 1), med, marker="o", zorder=3, c=[color_map[e] for e in order], edgecolor="black"
    )

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_title("Estancia UCI por etnia")
    ax.set_xlabel("Etnia")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(facecolor=color_map[e], edgecolor="black", label=e) for e in order]
    ax.legend(handles=handles, title="Etnia", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

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


def plot_dashboard_3x3(df: pd.DataFrame):
    fig, axs = plt.subplots(3, 3, figsize=(22, 15))
    axs = np.asarray(axs)

    plot_hist_los(df, ax=axs[0, 0])
    plot_box_by_gender(df, ax=axs[0, 1])
    plot_bar_admit_source(df, ax=axs[0, 2])

    plot_scatter_age_los(df, ax=axs[1, 0])
    plot_bar_mean_by_dx(df, ax=axs[1, 1])
    plot_bar_mean_by_hospital(df, ax=axs[1, 2])

    plot_violin_los_by_ethnicity(df, ax=axs[2, 0])
    plot_height_hist(df, ax=axs[2, 1])
    plot_pie_hospital_admit_source(df, ax=axs[2, 2])

    fig.tight_layout()
    return fig, axs

# -----------------------------
# Pacientes por región (usa hospital.csv)
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


def plot_patients_by_region(df_patient_with_region: pd.DataFrame, top_n: int | None = None):
    if "region" not in df_patient_with_region.columns:
        raise ValueError("El DataFrame no tiene la columna 'region'. Ejecuta primero el merge.")

    counts = df_patient_with_region["region"].fillna("Desconocido").value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    counts = counts.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index.astype(str), counts.values, edgecolor="black")
    ax.set_title("Pacientes por región")
    ax.set_xlabel("Número de pacientes")
    ax.set_ylabel("Región")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center")

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

@st.cache_resource(show_spinner=False)
def load_xgb_regressor(path: str):
    if path is None:
        return None
    # intenta joblib/pkl primero
    ext = os.path.splitext(path)[1].lower()
    if ext in (".joblib", ".pkl"):
        try:
            return joblib.load(path)
        except Exception:
            pass
    # fallback: formato nativo XGBoost
    if xgb is not None:
        try:
            model = xgb.XGBRegressor()
            model.load_model(path)
            return model
        except Exception:
            return None
    return None

@st.cache_resource(show_spinner=False)
def load_rf_regressor(path: str):
    if path is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def build_voting_regressor(xgb_model, rf_model, weights=(1, 3)):
    if (xgb_model is None) or (rf_model is None):
        return None
    return VotingRegressor([('xboost', xgb_model), ('rfreg', rf_model)], weights=list(weights))


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

    st.caption("También puedes subir archivos si no tienes acceso al directorio")
    up_patient = st.file_uploader("Subir patient.csv", type=["csv"], accept_multiple_files=False)
    up_hospital = st.file_uploader("Subir hospital.csv (opcional)", type=["csv"], accept_multiple_files=False)

    st.markdown("---")
    st.header("📊 Visualización")
    _viz_options = [
        "Resumen 3x3",
        "Estancia media UCI (Días)",
        "Estancia media UCI (Género)",
        "Admisión",
        "Estancia media UCI (Edad)",
        "Estancia media UCI (diagnosticos)",
        "Estancia media UCI",
        "Ethnicity",
        "Height",
        "Zona UCI",
        "Region",
    ]
    _selected = st.radio("Elige un gráfico", _viz_options, key="viz")

    st.header("🎛️ Parámetros de gráficos")
    bins_hist = st.slider("Bins histograma LOS", 10, 150, 50, step=5)
    logy_hist = st.checkbox("Escala log en histograma LOS", value=False)
    top_n = st.slider("Top N categorías (admisión/diagnóstico/hospital)", 3, 20, 8)
    trendline = st.checkbox("Línea de tendencia (Edad vs LOS)", value=True)
    method = st.selectbox("Método de tendencia", ["linear", "lowess"], index=0)
    add_kde = st.checkbox("KDE 2D (Edad vs LOS)", value=True)
    density_height = st.checkbox("Altura: normalizar (densidad)", value=False)

# -----------------------------
# Carga de datos
# -----------------------------

dfs = {}
if up_patient is not None:
    dfs["patient"] = pd.read_csv(up_patient)
    if up_hospital is not None:
        dfs["hospital"] = pd.read_csv(up_hospital)
else:
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

# Preprocesado principal
patient_raw = dfs["patient"]
df_patient = preprocess_patient(patient_raw)

# ------ Página: Predicciones ------
if st.session_state.get("_page") == "predicciones":
    st.title("🔮 Predicciones — Test set (VotingRegressor)")
    st.write(
        "Selecciona un paciente de la tabla de abajo. Con el índice de la fila sacamos la predicción "
        "de la estancia con el **model_VotingRegressor.joblib**."
    )

    # Rutas por defecto
    colp1, colp2 = st.columns(2)
    with colp1:
        path_xtest_csv = st.text_input("Ruta a X_test.csv", value="/app/app/src/test/X_test.csv")
        path_sparse = st.text_input("Ruta a X_T_test.npz (matriz dispersa)", value="/app/app/src/test/X_T_test.npz")
        path_ytest = st.text_input("Ruta a y_test.csv (opcional)", value="/app/app/src/test/y_test.csv")
    with colp2:
        model_path = st.text_input("Ruta a model_VotingRegressor.joblib", value="/app/app/src/models/model_VotingRegressor.joblib")
        up_model = st.file_uploader("Subir model_VotingRegressor.joblib (opcional)", type=["joblib", "pkl"])

    # Carga X_test.csv para indexar pacientes
    df_xtest = None
    try:
        df_xtest = pd.read_csv(path_xtest_csv)
    except Exception as e:
        st.error(f"No pude leer X_test.csv: {e}")

    # Carga matriz dispersa X_T_test
    X_T_test = None
    if os.path.exists(path_sparse):
        try:
            X_T_test = load_sparse_matrix(path_sparse)
        except Exception as e:
            st.error(f"No pude leer la matriz dispersa: {e}")
    else:
        st.warning(f"No existe el archivo: {path_sparse}")

    # Verificación de alineación filas
    if (df_xtest is not None) and (X_T_test is not None) and (X_T_test.shape[0] != len(df_xtest)):
        st.warning(f"Tamaño inconsistente entre X_test.csv (n={len(df_xtest)}) y X_T_test.npz (n={X_T_test.shape[0]}). Revisa el preprocesado.")

    # Selector de paciente
    idx = None
    if df_xtest is not None:
        st.subheader("👤 Selección de paciente")
        # intenta usar un ID si existe
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

    # Carga y_test si está disponible
    y_test = None
    if path_ytest and os.path.exists(path_ytest):
        try:
            y_test = pd.read_csv(path_ytest).squeeze()
        except Exception as e:
            st.warning(f"No pude leer y_test.csv: {e}")

    # Carga del VotingRegressor ya entrenado
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

    # Botones de acción
    colb1, colb2 = st.columns(2)
    do_pred_row = colb1.button("🎯 Predecir paciente seleccionado")
    do_pred_all = colb2.button("📈 Predecir y evaluar todo el test")

    # Predicción de una fila (manteniendo alineación con la matriz dispersa)
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
                    st.info(f"Ground truth (y_test) fila {row_idx}: **{gt:.3f}** (minutos)")
                    st.metric("Error absoluto", value=f"{abs(gt - y_hat):.3f}")
            except Exception as e:
                st.error(f"No se pudo predecir la fila {idx}: {e}")

    # Predicción y métricas en todo el test
    if do_pred_all:
        if X_T_test is None:
            st.error("Falta X_T_test para predecir el conjunto completo.")
        else:
            with st.spinner("Calculando predicciones en el test..."):
                try:
                    pred = model.predict(X_T_test)
                    st.success(f"Predicciones generadas: {len(pred):,}".replace(",", "."))
                    # métricas si hay y_test
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

                    # descarga CSV con predicciones (y GT si existe)
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

    st.caption(f"Modelo activo: {model_desc}")

    # Botones de acción
    colb1, colb2 = st.columns(2)
    do_pred_row = colb1.button("🎯 Predecir paciente seleccionado")
    do_pred_all = colb2.button("📈 Predecir y evaluar todo el test")

    # Predicción de una fila (manteniendo alineación con la matriz dispersa)
    if do_pred_row:
        if (X_T_test is None) or (df_xtest is None):
            st.error("Faltan X_T_test o X_test.csv para localizar la fila.")
        else:
            try:
                row_idx = int(idx)
                x_row = X_T_test.getrow(row_idx)
                y_hat = float(model_pred.predict(x_row)[0])
                st.success(f"Predicción para la fila {row_idx}: **{y_hat:.3f}** (minutos)")
                if y_test is not None and len(y_test) > row_idx:
                    gt = float(y_test.iloc[row_idx]) if hasattr(y_test, 'iloc') else float(y_test[row_idx])
                    st.info(f"Ground truth (y_test) fila {row_idx}: **{gt:.3f}** (minutos)")
                    st.metric("Error absoluto", value=f"{abs(gt - y_hat):.3f}")
            except Exception as e:
                st.error(f"No se pudo predecir la fila {idx}: {e}")

    # Predicción y métricas en todo el test
    if do_pred_all:
        if X_T_test is None:
            st.error("Falta X_T_test para predecir el conjunto completo.")
        else:
            with st.spinner("Calculando predicciones en el test..."):
                try:
                    pred = model_pred.predict(X_T_test)
                    st.success(f"Predicciones generadas: {len(pred):,}".replace(",", "."))
                    # métricas si hay y_test
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

                    # descarga CSV con predicciones (y GT si existe)
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

    # Botones de acción
    colb1, colb2 = st.columns(2)
    do_pred_row = colb1.button("🎯 Predecir paciente seleccionado")
    do_pred_all = colb2.button("📈 Predecir y evaluar todo el test")

    # Predicción de una fila (manteniendo alineación con la matriz dispersa)
    if do_pred_row:
        if (X_T_test is None) or (df_xtest is None):
            st.error("Faltan X_T_test o X_test.csv para localizar la fila.")
        else:
            try:
                row_idx = int(idx)
                x_row = X_T_test.getrow(row_idx)
                y_hat = float(votingreg.predict(x_row)[0])
                st.success(f"Predicción para la fila {row_idx}: **{y_hat:.3f}** (minutos)")
                if y_test is not None and len(y_test) > row_idx:
                    gt = float(y_test.iloc[row_idx]) if hasattr(y_test, 'iloc') else float(y_test[row_idx])
                    st.info(f"Ground truth (y_test) fila {row_idx}: **{gt:.3f}** (minutos)")
                    st.metric("Error absoluto", value=f"{abs(gt - y_hat):.3f}")
            except Exception as e:
                st.error(f"No se pudo predecir la fila {idx}: {e}")

    # Predicción y métricas en todo el test
    if do_pred_all:
        if X_T_test is None:
            st.error("Falta X_T_test para predecir el conjunto completo.")
        else:
            with st.spinner("Calculando predicciones en el test..."):
                try:
                    pred = votingreg.predict(X_T_test)
                    st.success(f"Predicciones generadas: {len(pred):,}".replace(",", "."))
                    # métricas si hay y_test
                    if y_test is not None and len(y_test) == len(pred):
                        try:
                            mae = mean_absolute_error(y_test, pred)
                            r2 = r2_score(y_test, pred)
                            st.write(f"**MAE test:** {mae:.4f}")
                            st.write(f"**R2 test:** {r2:.4f}")
                        except Exception as e:
                            st.warning(f"No pude calcular métricas: {e}")
                    elif y_test is not None:
                        st.warning(f"Dimensión inconsistente: y_test={len(y_test)} vs pred={len(pred)}. No calculo métricas.")

                    # descarga CSV con predicciones (y GT si existe)
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
    st.metric("LOS mediana", value=f"{df_patient['icu_los_days'].median():.1f} días")
with c4:
    st.metric("% Faltantes en LOS", value=f"{100*df_patient['icu_los_days'].isna().mean():.1f}%")

# Datos (opcional)
with st.expander("🔎 Ver primeras filas de patient"):
    st.dataframe(df_patient.head(50), use_container_width=True)

# === Visor simple de gráficos (sin tabs) ===

st.subheader("📈 Visualización actual")
_selected = st.session_state.get("viz", "Resumen 3x3")

if _selected == "Resumen 3x3":
    fig, _ = plot_dashboard_3x3(df_patient)
    render_plot(fig, "dashboard_3x3")
elif _selected == "Estancia media UCI (Días)":
    fig, _ = plot_hist_los(df_patient, bins=bins_hist, logy=logy_hist)
    render_plot(fig, "los_hist")
elif _selected == "Estancia media UCI (Género)":
    try:
        fig, _ = plot_box_by_gender(df_patient)
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
    fig, _ = plot_scatter_age_los(df_patient, trendline=trendline, method=method, add_kde=add_kde)
    render_plot(fig, "age_vs_los")
elif _selected == "Estancia media UCI (diagnosticos)":
    try:
        fig, _ = plot_bar_mean_by_dx(df_patient, top_n=top_n)
        render_plot(fig, "los_por_diagnostico")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Estancia media UCI":
    try:
        fig, _ = plot_bar_mean_by_hospital(df_patient, top_n=top_n)
        render_plot(fig, "los_por_hospital")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Ethnicity":
    fig, _ = plot_violin_los_by_ethnicity(df_patient, top_n=5, min_count=20)
    render_plot(fig, "los_por_etnia")
elif _selected == "Height":
    fig, _ = plot_height_hist(df_patient, bins=40, density=density_height, clip=(120, 210))
    render_plot(fig, "altura_hist_kde")
elif _selected == "Zona UCI":
    fig, _ = plot_pie_hospital_admit_source(df_patient, top_n=5)
    render_plot(fig, "admission_pie")
elif _selected == "Region":
    if "hospital" not in dfs:
        st.info("Sube `hospital.csv` o colócalo en el directorio de datos para ver este gráfico.")
    else:
        try:
            df_patient_region = merge_patient_with_region(df_patient, dfs["hospital"])
            fig, _ = plot_patients_by_region(df_patient_region)
            render_plot(fig, "pacientes_por_region")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico por región: {e}")

st.caption("Alejandro Cortijo Benito -- 2025 (alejandro.cortijo.benito@alumnos.upm.es)")