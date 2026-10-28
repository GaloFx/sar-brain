"""
Streamlit app: SAR espacial + grafo cerebral dinámico 3D
Compatible con Streamlit Community Cloud.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="SAR + Grafo Cerebral 3D (HCP DD)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 SAR Espacial + Grafo Cerebral Dinámico 3D")
st.caption(
    "Geometría real (fsaverage5 + Destrieux) · Fenotipo DD calibrado a distribución empírica HCP · "
    "Grosor y FC sintéticos (bloque sustituible)"
)


@st.cache_resource(show_spinner="Construyendo grafo cortical y estimando modelos SAR… (primera vez puede tardar 1–2 min)")
def load_state(seed: int = 20261011):
    from brain_sar import run_full_pipeline
    return run_full_pipeline(seed=int(seed))


# Sidebar
with st.sidebar:
    st.header("Controles de visualización")
    seed = st.number_input("Semilla reproducible", value=20261011, step=1)

    try:
        state = load_state(int(seed))
        load_ok = True
    except Exception as e:
        load_ok = False
        st.error(f"Error al cargar el pipeline: {type(e).__name__}: {e}")
        st.stop()

    n_frames = len(state["frame_times"])
    frame_idx = int(st.slider("Frame temporal", 0, max(n_frames - 1, 0), 0, 1))
    color_by = st.radio(
        "Colorear nodos por",
        options=["activity", "DD", "thickness"],
        format_func=lambda x: {
            "activity": "Actividad sintética",
            "DD": "Delay Discounting (HCP-scaled)",
            "thickness": "Grosor cortical",
        }[x],
        index=1,
    )
    n_edges = int(st.slider("Nº aristas funcionales", 20, 150, 60, 5))
    # Surface off by default on Cloud-friendly path (heavy mesh)
    show_surface = st.checkbox("Mostrar superficie cortical", False)
    show_spatial = st.checkbox("Mostrar capa espacial", True)
    show_functional = st.checkbox("Mostrar capa funcional", True)

    st.divider()
    st.markdown("**Mejor W (por AIC)**")
    st.success(
        f"{state['best_w']}  ·  ρ ≈ {float(state['summary_df'].loc[state['best_w'], 'rho']):.3f}"
    )
    st.caption(f"ρ verdadero (generador) = {state['rho_true']}")

tab_viz, tab_sar, tab_data, tab_audit = st.tabs(
    ["Visualización 3D", "Resultados SAR", "Datos & distribución", "Auditoría"]
)

with tab_viz:
    try:
        from viz import build_brain_figure

        fig = build_brain_figure(
            state,
            frame_idx=frame_idx,
            color_by=color_by,
            n_functional_edges=n_edges,
            show_surface=show_surface,
            show_spatial=show_spatial,
            show_functional=show_functional,
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Nodos: {state['N']}  ·  Aristas espaciales: {len(state['sp_i'])}  ·  "
            f"Frames: {n_frames}  ·  Fuente DD: `{state['dd_source']}`"
        )
    except Exception as e:
        st.error(f"Error al construir la figura 3D: {type(e).__name__}: {e}")
        st.exception(e)

with tab_sar:
    st.subheader("Diagnóstico espacial – Moran's I sobre DD")
    moran = state["moran_df"].copy()
    moran["sig"] = moran["p_sim"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    st.dataframe(moran, use_container_width=True)

    st.subheader("Estimación SAR (ML_Lag)")
    summary = state["summary_df"].copy()
    st.dataframe(summary, use_container_width=True)
    st.markdown(
        f"""
        **Interpretación (diseño sintético actual)**  
        - Mejor especificación: **{state['best_w']}** (mínimo AIC).  
        - ρ estimado ≈ **{float(summary.loc[state['best_w'], 'rho']):.3f}** (verdadero = {state['rho_true']}).  
        - β_grosor negativo → mayor grosor se asocia a menor DD (tras controlar dependencia espacial).  
        - β_conectividad positivo → mayor FC local se asocia a mayor DD en este diseño.
        """
    )

with tab_data:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución DD (HCP real / fallback)")
        st.write(
            pd.Series(state["dd_real"], name="DDisc_SV_1mo_40K")
            .describe()
            .round(1)
            .to_frame()
        )
        st.caption(f"Fuente: `{state['dd_source']}` · N sujetos = {len(state['dd_real'])}")
    with col2:
        st.subheader("Variables a nivel de nodo")
        st.write(
            state["df"][["cortical_thickness", "functional_conn", "DD"]]
            .describe()
            .round(3)
        )
        st.caption(f"N nodos corticales = {state['N']}")

    st.subheader("Nodos corticales (primeras filas)")
    st.dataframe(state["nodes_df"].head(15), use_container_width=True)

    st.download_button(
        "Descargar variables_SAR.csv",
        data=state["df"].to_csv(index=False).encode("utf-8"),
        file_name="variables_SAR.csv",
        mime="text/csv",
    )
    st.download_button(
        "Descargar resultados_SAR.csv",
        data=state["summary_df"].to_csv().encode("utf-8"),
        file_name="resultados_SAR.csv",
        mime="text/csv",
    )

with tab_audit:
    st.markdown(
        """
### Auditoría del pipeline (v1.2 – Streamlit Cloud)

#### Correcciones v1.2
- API Plotly 5/6: eliminados `titlefont` / `backgroundcolor` incompatibles.
- Malla cortical submuestreada (1/8 caras) y **superficie desactivada por defecto** (activar en sidebar).
- Arrays convertidos a `list`/`float` nativos para serialización estable.
- Índices de slider forzados a `int`.
- Mensajes de error visibles (no redactados) en la pestaña de visualización.
- `requirements.txt` con versiones acotadas.

#### ✅ Cumple
| Aspecto | Estado |
|---------|--------|
| Geometría cortical real (nilearn fsaverage5 + Destrieux) | ✅ |
| Fenotipo DD usa distribución empírica real del CSV HCP | ✅ |
| Matrices W row-estandarizadas; estimación ML_Lag | ✅ |
| Sin islas en W (vecinos garantizados) | ✅ |
| Código libre de dependencias Colab / Drive | ✅ |
| Interfaz Streamlit Cloud-ready | ✅ |

#### ⚠️ Limitaciones
- Grosor cortical y FC son **sintéticos** (bloque en `brain_sar.generate_node_features`).
- SAR a nivel de **nodos/regiones**, no de sujetos.
- Primera ejecución descarga datos nilearn (requiere red; ~1–2 min en Cloud).

#### Cómo sustituir por datos reales (HCP / BALSA)
1. Matrices de grosor medio por región Destrieux.
2. Matrices de FC (Fisher-z).
3. Sustituir vectores en `generate_node_features` dentro de `brain_sar.py`.
        """
    )
