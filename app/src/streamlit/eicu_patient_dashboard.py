from __future__ import annotations

import os
import io
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
import geopandas as gpd
import unicodedata

matplotlib.use("Agg")  # backend no interactivo para Streamlit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

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
    c_path = os.path.join(data_dir, "carePlanGoal.csv")

    if os.path.exists(p_path):
        dfs["patient"] = read_csv_safe(p_path)
    if os.path.exists(h_path):
        dfs["hospital"] = read_csv_safe(h_path)
    if os.path.exists(c_path):
        dfs["carePlanGoal"] = read_csv_safe(c_path)
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

def plot_hist_los(
    df: pd.DataFrame,
    bins: int = 50,
    logy: bool = False,
    ax=None,
    line: bool = False,
    show_mean: bool = False,
    show_median: bool = False,
):
    """
    Histograma de estancia en UCI (días).
    - line=True añade curva de densidad (tendencia)
    - show_mean/median controlan si se muestran las líneas y anotaciones
    - Leyenda adaptativa según los elementos visibles
    """
    if "icu_los_days" not in df.columns:
        raise ValueError("Falta la columna 'icu_los_days'.")

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    x = pd.to_numeric(df["icu_los_days"], errors="coerce").dropna()
    if len(x) == 0:
        ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center", transform=ax.transAxes)
        if created:
            fig.tight_layout()
        return fig, ax

    # Histograma
    density = bool(line)
    counts, edges, _ = ax.hist(
        x,
        bins=bins,
        edgecolor="black",
        alpha=0.65,
        color="#5DADE2",
        density=density,
        linewidth=0.8,
        label="Frecuencia" if not density else "Densidad",
    )

    # KDE opcional
    handles, labels = ax.get_legend_handles_labels()
    if line:
        sns.kdeplot(
            x=x,
            ax=ax,
            linewidth=2,
            color="#1A5276",
            bw_method="scott",
            cut=0,
            label="Tendencia",
        )
        ax.set_ylabel("Densidad")
    else:
        ax.set_ylabel("Frecuencia")

    ymax = counts.max() * (1.1 if not density else 1.3)

    # Media
    if show_mean:
        mean_val = x.mean()
        ax.axvline(mean_val, color="#E74C3C", linestyle="--", linewidth=2, label=f"Media: {mean_val:.1f}")
        ax.annotate(
            f"Media = {mean_val:.1f} días",
            xy=(mean_val, ymax * 0.55),
            xytext=(mean_val + (x.max() * 0.05), ymax * 0.55),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#E74C3C"),
            fontsize=9,
            color="#E74C3C",
            ha="left",
            va="center",
        )

    # Mediana
    if show_median:
        median_val = x.median()
        ax.axvline(median_val, color="#27AE60", linestyle="-.", linewidth=2, label=f"Mediana: {median_val:.1f}")
        ax.annotate(
            f"Mediana = {median_val:.1f} días",
            xy=(median_val, ymax * 0.75),
            xytext=(median_val + (x.max() * 0.05), ymax * 0.75),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#27AE60"),
            fontsize=9,
            color="#27AE60",
            ha="left",
            va="center",
        )

    # Estilo visual limpio
    ax.set_title("Distribución de la estancia en UCI", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Estancia (días)", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_facecolor("white")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.6)
    ax.spines["bottom"].set_alpha(0.6)

    # Leyenda solo con elementos activos
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper right")

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
    Scatter de edad (x) vs estancia en UCI (y) con:
      - Clustering KMeans coloreado por cluster.
      - Línea vertical en la mediana o media de edad.
      - Líneas divisorias entre clusters y centroides marcados.
      - Valor de la línea mostrado a la derecha.
    """
    if not {"age_years", "icu_los_days"} <= set(df.columns):
        raise ValueError("Faltan columnas requeridas: 'age_years' y 'icu_los_days'.")

    tmp = df[["age_years", "icu_los_days"]].dropna()
    if len(tmp) == 0:
        raise ValueError("No hay datos válidos tras eliminar NaN.")

    if len(tmp) > max_points:
        tmp = tmp.sample(max_points, random_state=42)

    n_clusters = max(1, min(n_clusters, len(tmp)))

    # --- KMeans ---
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(tmp[["age_years", "icu_los_days"]].to_numpy())
    centers = km.cluster_centers_

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    # --- Colores y símbolos ---
    palette = sns.color_palette("Set2", n_colors=n_clusters)
    roman_map = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

    # --- Dibujar regiones de cluster (contornos suaves) ---
    x_min, x_max = tmp["age_years"].min() - 2, tmp["age_years"].max() + 2
    y_min, y_max = tmp["icu_los_days"].min() - 1, tmp["icu_los_days"].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Fondo y líneas divisorias
    ax.contourf(xx, yy, Z, alpha=0.08, cmap="Set2", levels=n_clusters)
    ax.contour(xx, yy, Z, colors="gray", linewidths=0.8, linestyles="--", alpha=0.6)

    # --- Puntos por cluster ---
    for k in range(n_clusters):
        sub = tmp[labels == k]
        ax.scatter(
            sub["age_years"],
            sub["icu_los_days"],
            s=16,
            alpha=0.55,
            color=palette[k],
            label=f"Cluster {roman_map.get(k + 1, str(k + 1))}",
            zorder=2,
        )

    # --- Centroides ---
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        color="black",
        marker="X",
        s=120,
        edgecolor="white",
        linewidth=1.2,
        label="Centroides",
        zorder=4,
    )

    # --- Línea vertical (media o mediana) con valor ---
    if line in ("median", "mean"):
        xval = tmp["age_years"].median() if line == "median" else tmp["age_years"].mean()
        label_text = f"{'Mediana' if line == 'median' else 'Media'} edad"
        ax.axvline(
            xval,
            linestyle="--",
            linewidth=2,
            color="black",
            label=label_text,
            zorder=3,
        )

        # Mostrar el valor numérico a la derecha de la línea
        y_text = tmp["icu_los_days"].max() * 0.95  # altura relativa
        ax.text(
            xval + (tmp["age_years"].max() - tmp["age_years"].min()) * 0.01,  # leve desplazamiento a la derecha
            y_text,
            f"{xval:.1f} años",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="black",
            weight="bold",
            backgroundcolor="white",
            zorder=5,
        )

    # --- Estilo y formato ---
    ax.set_title("Edad vs estancia en UCI (clusters)", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Estancia UCI (días)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="best", fontsize=9)

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

    ax.set_title("Estancia en UCI por género", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Género")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(False)  

    if created:
        fig.tight_layout()
    return fig, ax

def plot_bar_mean_by_dx(
    df: pd.DataFrame,
    top_n: int = 8,
    hue_col: str | None = "apacheadmissiondx",
    ax=None,
    annotate: bool = True,
):
    """
    Muestra la estancia media en UCI por diagnóstico (apacheadmissiondx),
    coloreando cada barra con un color distinto y sin cuadrícula.
    """
    if "apacheadmissiondx" not in df.columns:
        raise ValueError("Falta la columna 'apacheadmissiondx'.")

    data = df.copy()
    data["apacheadmissiondx"] = data["apacheadmissiondx"].fillna("Desconocido")

    if hue_col is not None:
        if hue_col not in data.columns:
            raise ValueError(f"Falta la columna de hue '{hue_col}'.")
        data[hue_col] = data[hue_col].fillna("Desconocido")

    # Seleccionar top diagnósticos más frecuentes
    top_categories = (
        data["apacheadmissiondx"].value_counts().head(top_n).index.tolist()
    )
    data = data[data["apacheadmissiondx"].isin(top_categories)]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
        created = True
    else:
        fig = ax.figure

    # Calcular estancia media y preparar DataFrame
    plot_df = (
        data.groupby("apacheadmissiondx", as_index=False)["icu_los_days"]
        .mean()
        .rename(columns={"icu_los_days": "mean_los"})
    )
    plot_df = plot_df.sort_values("mean_los", ascending=False)

    # Dibujar barras
    sns.barplot(
        data=plot_df,
        x="apacheadmissiondx",
        y="mean_los",
        hue=hue_col,
        order=plot_df["apacheadmissiondx"],
        dodge=False,
        ax=ax,
    )

    # Estilo limpio (sin grid)
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.6)
    ax.spines["bottom"].set_alpha(0.6)

    # Títulos y ejes
    ax.set_title(f"Estancia media en UCI por diagnóstico (Top {top_n})", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Diagnóstico de admisión", fontsize=11)
    ax.set_ylabel("Estancia media (días)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    # Anotaciones opcionales
    if annotate:
        try:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
        except Exception:
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(
                    f"{h:.1f}",
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
):
    """
    Boxplot de estancia en UCI (icu_los_days) por etnia (ethnicity).
    - Filtra a las 'top_n' etnias más frecuentes y descarta grupos con < min_count.
    - Ordena las categorías por mediana de estancia (descendente).
    - Colorea por categoría y añade leyenda consistente (sin recuadro).
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
        ax.text(
            0.5, 0.5, "Sin datos suficientes tras el filtrado",
            ha="center", va="center", transform=ax.transAxes
        )
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

    ax.set_title("Estancia UCI por etnia", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Etnia")
    ax.set_ylabel("Estancia UCI (días)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(order, rotation=25, ha="right")

    # --- Leyenda sin recuadro ---
    handles = [
        mpatches.Patch(facecolor=color_map[e], edgecolor="black", label=e)
        for e in order
    ]
    ax.legend(
        handles=handles,
        title="Etnia",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,  
    )

    if created:
        fig.tight_layout()
    return fig, ax

def plot_height_hist(
    df: pd.DataFrame,
    ax=None,
    bins: int = 40,
    density: bool = False,
    clip: tuple | None = (120, 210),
    show_mean: bool = False,  
):
    """
    Histograma de altura de admisión (cm) con KDE y línea opcional de media.
    - Quita el marco (spines) y usa estilo limpio.
    - Muestra distribución por grupos (<170, 170–180, ≥180).
    - Si show_mean=True, dibuja línea vertical con el valor medio.
    """
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    # Altura en cm
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
    bin_width = float(np.diff(edges).mean())

    tmp = pd.DataFrame({"height_cm": height_cm})
    bins_groups = [-np.inf, 170, 180, np.inf]
    labels = ["<170", "170–180", "≥180"]
    tmp["height_group"] = pd.cut(tmp["height_cm"], bins=bins_groups, labels=labels, right=False)

    palette = {"<170": "#1f77b4", "170–180": "#ff7f0e", "≥180": "#2ca02c"}

    multiple = "stack" if not density else "layer"
    stat = "density" if density else "count"

    # --- Histograma principal ---
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

    # --- KDE en negro ---
    sns.kdeplot(height_cm.values, ax=ax, bw_method="scott", cut=0, color="black", linewidth=2)

    # Ajustar KDE si no es densidad
    if not density:
        line = ax.lines[-1]
        y = line.get_ydata()
        y_counts = y * len(height_cm) * bin_width
        line.set_ydata(y_counts)

    # --- Línea y anotación de la media (opcional) ---
    if show_mean:
        mean_val = height_cm.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Media: {mean_val:.1f} cm")

        x_offset = (height_cm.max() - height_cm.min()) * 0.02
        ymax = ax.get_ylim()[1]

        ax.text(
            mean_val + x_offset,
            ymax * 0.9,
            f"Media\n{mean_val:.1f} cm",
            color="red",
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
        )

    # --- Leyenda de grupos ---
    present = [g for g in labels if g in tmp["height_group"].dropna().unique().tolist()]
    handles = [Patch(facecolor=palette[g], edgecolor="black", label=g) for g in present]

    # Añadir media a la leyenda solo si se muestra
    if show_mean:
        mean_val = height_cm.mean()
        handles.append(Patch(facecolor="none", edgecolor="red", label=f"Media: {mean_val:.1f} cm"))

    ax.legend(
        handles=handles,
        title="Altura (cm)",
        frameon=False,
        loc="best",
        fontsize=9,
    )

    # --- Estilo limpio ---
    ax.set_title("Distribución de altura de admisión (cm)", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Densidad" if density else "Frecuencia")

    ax.grid(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("white")

    if created:
        fig.tight_layout()
    return fig, ax

def plot_pie_hospital_admit_source(
    df: pd.DataFrame,
    top_n: int = 5,
    ax=None,
    show_counts: bool = True,
    label_radius: float = 1.28,
    line_radius: float = 0.85,
    min_sep: float = 0.125,
    clip_low: float = -0.125,
    clip_high: float = 0.90,
    title_y: float = 1.06,
    top_adjust: float = 0.88,
):
    """
    Pie chart de fuentes de admisión hospitalaria equilibrado visualmente.
    - Las flechas parten del centro del pastel.
    - Las etiquetas se superponen sobre las líneas.
    - El ángulo se ajusta automáticamente para balancear lados.
    """
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

    # === Ángulo de inicio balanceado ===
    # Buscar el ángulo del sector más grande y colocarlo centrado abajo
    largest_idx = int(np.argmax(vals))

    cumulative = np.cumsum(vals) / total * 360
    center_angle = (cumulative[largest_idx] - vals[largest_idx] / total * 180)
    startangle = 270 - center_angle  

    wedges, _ = ax.pie(vals, startangle=startangle, counterclock=False, labels=None, autopct=None)
    ax.axis("equal")

    # --- función de separación vertical sin solapamientos ---
    def _spread_on_side(items, min_sep, low, high):
        if not items:
            return
        items.sort(key=lambda d: d["y"])
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
        item = {"theta": theta, "x": x, "y": y, "lab": lab, "pct": pct, "n": int(n), "color": wedge.get_facecolor()}
        (right_items if x >= 0 else left_items).append(item)

    _spread_on_side(right_items, min_sep=min_sep, low=clip_low, high=clip_high)
    _spread_on_side(left_items, min_sep=min_sep, low=clip_low, high=clip_high)

    n_labels = len(labels)
    rad_curve = 0.25 if n_labels <= 6 else 0.15 if n_labels <= 10 else 0.08

    # --- dibujar anotaciones ---
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

            # Flecha (debajo)
            ax.annotate(
                "",
                xytext=(tx, ty),
                xy=xy,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=1.1,
                    color="black",
                    alpha=0.8,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle=f"arc3,rad={rad_curve}",
                    zorder=1,
                ),
            )

            # Texto (encima, con borde blanco)
            text = ax.text(
                tx,
                ty,
                txt,
                ha=ha,
                va="center",
                fontsize=9,
                color="black",
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            )
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

    ax.set_title(
        f"Distribución de admisión hospitalaria (Top {top_n})",
        y=title_y,
        fontsize=13,
        weight="bold",
    )

    if created:
        fig.subplots_adjust(top=top_adjust)
    return fig, ax

def plot_dashboard_3x3(df, df_h):
    fig, axs = plt.subplots(3, 3, figsize=(25, 18))
    axs = np.asarray(axs)

    plot_hist_los(df, line=density_hist, bins=bins_hist, show_mean=mean_hist, show_median=median_hist, ax=axs[0, 0])
    plot_violin_by_gender(df, ax=axs[0, 1])
    plot_pie_hospital_admit_source(df, top_n=top_n_zone, ax=axs[0, 2])

    plot_scatter_age_los(df, n_clusters=cluster_n, line=trendline, ax=axs[1, 0])
    plot_bar_mean_by_dx(df, top_n=top_n_diag, ax=axs[1, 1])
    plot_patients_by_region(df_h, ax=axs[1, 2])

    plot_box_los_by_ethnicity(df, top_n=5, min_count=20, showfliers=showfliers_boxplot, ax=axs[2, 0])
    plot_height_hist(df, bins=40, density=density_height, show_mean=show_mean_height, clip=(120, 210), ax=axs[2, 1])
    plot_careplan_counts(carPlan_raw, top_n=top_n_care, ax=axs[2, 2])

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

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

def plot_patients_by_region(df_patient_with_region: pd.DataFrame, ax=None, json_path: str = "/app/app/db/us-states.json"):
    """
    Mapa de EE.UU. mostrando pacientes por región (Northeast, South, West, Midwest, _Unknown).
    - Usa un GeoJSON local (json_path).
    - Colorea cada región y muestra porcentajes.
    - Normaliza nombres (_Unkhown -> _Unknown).
    """

    if "region" not in df_patient_with_region.columns:
        raise ValueError("El DataFrame no tiene la columna 'region'.")

    # --- Normalizar nombres de región ---
    raw = df_patient_with_region["region"].astype(str).map(_norm).str.lower()
    mapping = {
        "_unknown": "_Unknown",
        "_unkhown": "_Unknown",
        "unknown": "_Unknown",
        "desconocido": "_Unknown",
        "northeast": "Northeast",
        "midwest": "Midwest",
        "south": "South",
        "west": "West",
    }
    df_patient_with_region = df_patient_with_region.copy()
    df_patient_with_region["region"] = raw.map(mapping).fillna("_Unknown")

    # --- Contar pacientes por región ---
    counts = df_patient_with_region["region"].value_counts()
    total = int(counts.sum())
    percents = (100 * counts / total).round(1)

    # --- Cargar el GeoJSON local ---
    states = gpd.read_file(json_path)

    # --- Asignar regiones (basado en U.S. Census Bureau) ---
    northeast = {"Maine","New Hampshire","Vermont","Massachusetts","Rhode Island","Connecticut",
                 "New York","New Jersey","Pennsylvania"}
    midwest = {"Ohio","Indiana","Illinois","Michigan","Wisconsin",
               "Minnesota","Iowa","Missouri","North Dakota","South Dakota","Nebraska","Kansas"}
    south = {"Delaware","Maryland","District of Columbia","Virginia","West Virginia",
             "North Carolina","South Carolina","Georgia","Florida",
             "Kentucky","Tennessee","Alabama","Mississippi",
             "Arkansas","Louisiana","Oklahoma","Texas"}
    west = {"Montana","Idaho","Wyoming","Colorado","New Mexico",
            "Arizona","Utah","Nevada","Washington","Oregon","California","Alaska","Hawaii"}

    def assign_region(state):
        if state in northeast:
            return "Northeast"
        elif state in midwest:
            return "Midwest"
        elif state in south:
            return "South"
        elif state in west:
            return "West"
        else:
            return "_Unknown"

    states["region"] = states["name"].apply(assign_region)

    # --- Paleta y orden de leyenda ---
    palette = {
        "Northeast": "#1f77b4",
        "Midwest": "#2ca02c",
        "South": "#ff7f0e",
        "West": "#9467bd",
        "_Unknown": "#d3d3d3",
    }
    legend_order = ["Northeast", "Midwest", "South", "West", "_Unknown"]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created = True
    else:
        fig = ax.figure

    # --- Dibujar mapa ---
    for region in legend_order:
        color = palette[region]
        states[states["region"] == region].plot(
            ax=ax, color=color, edgecolor="white", linewidth=0.8, alpha=0.9
        )

    # --- Título y estilo ---
    ax.set_title(
        "Distribución de pacientes por región de EE.UU.",
        fontsize=13, weight="bold", pad=10
    )
    ax.axis("off")

    # --- Leyenda con colores + porcentaje ---
    legend_elements = []
    for region in legend_order:
        pct = float(percents.get(region, 0.0))
        label = f"{region} ({pct:.2f}%)"
        patch = Patch(facecolor=palette[region], edgecolor="black", label=label)
        legend_elements.append(patch)

    ax.legend(
        handles=legend_elements,
        title="Región EE.UU.",
        frameon=False,
        loc="lower left",
        fontsize=9,
    )

    if created:
        fig.tight_layout()
    return fig, ax

# -----------------------------
# Global Care UCI (carePlanGoal.csv)
# -----------------------------

def plot_careplan_counts(
    df: pd.DataFrame,
    ax=None,
    top_n: int | None = None,
    min_count: int = 1,
    title: str = "Frecuencia de Planes de Cuidado (Care Plans)",
):
    """
    Gráfico de barras de frecuencia de aparición de Care Plans.
    
    - Cuenta cuántos pacientes tienen valor > 0 por tipo de Care Plan.
    - Elimina el prefijo 'Care_' de los nombres de columna.
    - Permite filtrar por mínimo conteo (min_count) o limitar a los top N.
    - Devuelve fig, ax al estilo plot_height_hist.
    """

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created = True
    else:
        fig = ax.figure

    # --- Validar columnas ---
    care_cols = [c for c in df.columns if c != "patientunitstayid"]
    if not care_cols:
        ax.text(0.5, 0.5, "Sin columnas válidas", ha="center", va="center", transform=ax.transAxes)
        if created:
            fig.tight_layout()
        return fig, ax

    # --- Calcular conteos ---
    care_counts = (df[care_cols] > 0).sum().reset_index()
    care_counts.columns = ["Care_Plan", "Count"]
    care_counts["Care_Plan"] = care_counts["Care_Plan"].str.replace("^Care_", "", regex=True)

    # --- Filtrado por mínimo o top N ---
    care_counts = care_counts[care_counts["Count"] >= min_count]
    care_counts = care_counts.sort_values("Count", ascending=False)
    if top_n is not None:
        care_counts = care_counts.head(top_n)

    if care_counts.empty:
        ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center", transform=ax.transAxes)
        if created:
            fig.tight_layout()
        return fig, ax

    # --- Gráfico ---
    palette = sns.color_palette("Blues", n_colors=len(care_counts))
    bars = ax.barh(care_counts["Care_Plan"], care_counts["Count"], color=palette, edgecolor="black", alpha=0.8)
    ax.invert_yaxis()

    # Etiquetas sobre las barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(care_counts["Count"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(width)}", va="center", fontsize=9, weight="bold")

    # --- Estilo limpio ---
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Número de pacientes con registro (>0)")
    ax.set_ylabel("Tipo de Care Plan")
    ax.grid(False)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("white")

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

    data_dir = "/app/app/db/csv_clean"
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
        "Especialidad UCI (Tratamiento)",
    ]
    _selected = st.radio("Elige un gráfico", _viz_options, key="viz")

    st.header("🎛️ Parámetros de gráficos")

    st.markdown("**Estancia media UCI (Días)**", unsafe_allow_html=True)
    
    bins_hist = st.slider("Bins histograma:", 10, 150, 50, step=5)

    col_plot1, col_plot2, col_plot3 = st.columns(3)
    with col_plot1:
        density_hist = st.checkbox("Tendencia", value=False)
    with col_plot2:
        mean_hist = st.checkbox("Media", value=True)
    with col_plot3:
        median_hist = st.checkbox("Mediana", value=True)

    st.markdown("**Admisión**", unsafe_allow_html=True)
    
    with st.sidebar:
        top_n_zone = st.number_input(
            "Top N zonas admisiones:",
            min_value=2,
            max_value=8,
            value=5,
            step=1,
            help="Número de zonas a mostrar en el pie. (max 8)"
        )

    st.markdown("**Estancia media UCI (Edad)**", unsafe_allow_html=True)

    cluster_n = st.slider("N clusters", 2, 5, 3) 
    opcion = st.selectbox("Añadir tendencia:", ["Ninguna", "Mediana", "Media"], index=0)
    trendline = {"Ninguna": None, "Mediana": "median", "Media": "mean"}[opcion]

    st.markdown("**Estancia media UCI (Diagnosticos)**", unsafe_allow_html=True)

    top_n_diag = st.slider("Top N categorías diagnosticos.", 3, 10, 8)

    st.markdown("**Ethnicity**", unsafe_allow_html=True)

    showfliers_boxplot = st.radio(
        "Mostrar valores atípicos en boxplots",
        ["No", "Sí"],
        index=0,
        horizontal=True,
    ) == "Sí"

    st.markdown("**Height**", unsafe_allow_html=True)

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        density_height = st.toggle("Normalizar altura", value=False)
    with col_plot2:
        show_mean_height = st.toggle("Media altura", value=False)

    st.markdown("**Especialidad UCI (Tratamiento)**", unsafe_allow_html=True)

    top_n_care = st.radio(
        "Top N especialidades:",
        options=[2, 3, 4, 5, 6],
        index=2,
        horizontal=True,
        help="Número de especialidades a mostrar. (max 6)"
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
carPlan_raw = dfs["carePlanGoal"]
df_patient = preprocess_patient(patient_raw)

# ------ Página: Predicciones ------
if st.session_state.get("_page") == "predicciones":
    st.title("🔮 Predicciones — Test set (VotingRegressor)")
    st.write(
        "Selecciona un paciente de la tabla de abajo. Con el índice de la fila sacamos la predicción "
        "de su estancia en UCI con el **model_VotingRegressor.joblib**."
    )

    path_xtest_csv = "/app/app/src/test/X_test.csv"
    path_sparse = "/app/app/src/test/X_T_test.npz"
    path_ytest = "/app/app/src/test/y_test.csv"
    model_path = "/app/app/src/models/model_VotingRegressor.joblib"

    # 🔹 Solo mostramos el uploader del modelo
    st.subheader("📦 Cargar modelo")
    up_model = st.file_uploader(
        "Subir model_VotingRegressor.joblib (opcional)",
        type=["joblib", "pkl"],
        help="Si se proporciona, se usará este modelo en lugar del guardado por defecto."
    )

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
            idx = st.number_input("Selecciona una fila (0-index) del conjunto de test para hacer inferencia", min_value=0, max_value=max(0, len(df_xtest)-1), value=0, step=1)

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
                    st.metric("Error absoluto", value=f"{abs(gt - y_hat):.3f} minutos")
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
                            st.write(f"**MAE test:** {mae:.4f} minutos")
                            st.write(f"**R² test:** {r2:.4f} minutos")
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
    fig, _ = plot_hist_los(df_patient, line=density_hist, bins=bins_hist, show_mean=mean_hist, show_median=median_hist)
    render_plot(fig, "los_hist")
elif _selected == "Estancia media UCI (Género)":
    try:
        fig, _ = plot_violin_by_gender(df_patient)
        render_plot(fig, "los_box_genero")
    except Exception as e:
        st.warning(f"No se pudo generar el boxplot: {e}")
elif _selected == "Admisión":
    try:
        fig, _ = plot_pie_hospital_admit_source(df_patient, top_n=top_n_zone)
        render_plot(fig, "admission_pie")
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
        fig, _ = plot_patients_by_region(df_patient_region)
        render_plot(fig, "los_por_hospital")
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")
elif _selected == "Ethnicity":
    fig, _ = plot_box_los_by_ethnicity(df_patient, showfliers=showfliers_boxplot, top_n=5, min_count=20)
    render_plot(fig, "los_por_etnia")
elif _selected == "Height":
    fig, _ = plot_height_hist(df_patient, bins=40, density=density_height, show_mean=show_mean_height, clip=(120, 210))
    render_plot(fig, "altura_hist_kde")
elif _selected == "Especialidad UCI (Tratamiento)":
    fig, _ = plot_careplan_counts(carPlan_raw, top_n=top_n_care)
    render_plot(fig, "careplan_counts")

st.caption("Alejandro Cortijo Benito -- 2025 (alejandro.cortijo.benito@alumnos.upm.es)")