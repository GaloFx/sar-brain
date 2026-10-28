"""Plotly 3D brain graph visualization helpers (Streamlit Cloud compatible)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def edge_xyz(ii, jj, positions):
    """Build interleaved xyz for line segments (None breaks)."""
    xx, yy, zz = [], [], []
    for a, b in zip(ii, jj):
        xx.extend([float(positions[a, 0]), float(positions[b, 0]), None])
        yy.extend([float(positions[a, 1]), float(positions[b, 1]), None])
        zz.extend([float(positions[a, 2]), float(positions[b, 2]), None])
    return xx, yy, zz


def make_functional_trace(C, threshold_count, positions, opacity=0.75):
    iu, ju = np.triu_indices_from(C, k=1)
    vals = np.asarray(C[iu, ju], dtype=float)
    n_keep = min(int(threshold_count), len(vals))
    if n_keep <= 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Conectividad funcional")
    valid = np.argsort(np.abs(vals))[-n_keep:]
    ii, jj = iu[valid], ju[valid]
    xx, yy, zz = edge_xyz(ii, jj, positions)
    return go.Scatter3d(
        x=xx,
        y=yy,
        z=zz,
        mode="lines",
        line=dict(color=f"rgba(230,50,40,{opacity})", width=3.0),
        name="Conectividad funcional",
        hoverinfo="skip",
    )


def build_brain_figure(
    state: dict,
    frame_idx: int = 0,
    color_by: str = "activity",
    n_functional_edges: int = 80,
    show_surface: bool = True,
    show_spatial: bool = True,
    show_functional: bool = True,
    height: int = 680,
):
    """Build interactive Plotly 3D figure from pipeline state.

    Written to be compatible with Plotly 5.x and 6.x (Streamlit Cloud).
    """
    coords = np.asarray(state["coords"], dtype=float)
    coords_hemi = state["coords_hemi"]
    faces_hemi = state["faces_hemi"]
    y = np.asarray(state["y"], dtype=float)
    cortical_thickness = np.asarray(state["cortical_thickness"], dtype=float)
    activity_frames = state["activity_frames"]
    corr_frames = state["corr_frames"]
    frame_times = state["frame_times"]
    nodes_df = state["nodes_df"]
    sp_i = np.asarray(state["sp_i"])
    sp_j = np.asarray(state["sp_j"])

    # Harden index types (Streamlit sliders can return numpy/float)
    frame_idx = int(np.clip(int(frame_idx), 0, len(frame_times) - 1))
    n_functional_edges = int(n_functional_edges)

    fig = go.Figure()

    # --- Surface (heavy): aggressive subsample for Cloud memory/time limits ---
    if show_surface:
        for hemi in ("left", "right"):
            xyz = np.asarray(coords_hemi[hemi], dtype=float)
            faces = np.asarray(faces_hemi[hemi], dtype=int)
            # Keep ~1/8 of faces
            faces = faces[::8]
            fig.add_trace(
                go.Mesh3d(
                    x=xyz[:, 0].tolist(),
                    y=xyz[:, 1].tolist(),
                    z=xyz[:, 2].tolist(),
                    i=faces[:, 0].tolist(),
                    j=faces[:, 1].tolist(),
                    k=faces[:, 2].tolist(),
                    color="#E2E7EC",
                    opacity=0.22,
                    name=f"Superficie {hemi}",
                    hoverinfo="skip",
                    flatshading=True,
                    showscale=False,
                )
            )

    # --- Spatial edges ---
    if show_spatial and len(sp_i) > 0:
        sx, sy, sz = edge_xyz(sp_i, sp_j, coords)
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color="rgba(70,100,140,0.25)", width=1.5),
                name="Capa espacial",
                hoverinfo="skip",
            )
        )

    # --- Node colors ---
    if color_by == "DD":
        node_color = y
        cmin = float(np.nanpercentile(y, 2))
        cmax = float(np.nanpercentile(y, 98))
        cbar_title = "DD (HCP-scaled)"
    elif color_by == "thickness":
        node_color = cortical_thickness
        cmin, cmax = 1.8, 3.2
        cbar_title = "Grosor (mm)"
    else:
        node_color = np.asarray(activity_frames[frame_idx], dtype=float)
        cmin, cmax = -1.0, 1.0
        cbar_title = "Actividad"

    hover_text = [
        (
            f"Nodo {int(r.node)} | {r.hemisphere}<br>"
            f"Parcela {int(r.parcel_id)}<br>"
            f"DD={float(y[int(r.node)]):.1f} | "
            f"Grosor={float(cortical_thickness[int(r.node)]):.2f}"
        )
        for r in nodes_df.itertuples()
    ]

    # Colorbar: plain string title for Plotly 5/6 compatibility
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            z=coords[:, 2].tolist(),
            mode="markers",
            marker=dict(
                size=6,
                color=np.asarray(node_color, dtype=float).tolist(),
                colorscale="RdYlBu_r",
                cmin=float(cmin),
                cmax=float(cmax),
                opacity=0.95,
                colorbar=dict(title=cbar_title, x=1.02, len=0.65),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Regiones",
        )
    )

    # --- Functional edges ---
    if show_functional:
        fig.add_trace(
            make_functional_trace(
                np.asarray(corr_frames[frame_idx], dtype=float),
                n_functional_edges,
                coords,
            )
        )

    t_val = int(frame_times[frame_idx])
    fig.update_layout(
        title=f"Grafo cerebral 3D | t={t_val} | color={color_by}",
        template="plotly_white",
        height=int(height),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#FFFFFF",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
        ),
        legend=dict(x=0.01, y=0.99),
        showlegend=True,
    )
    return fig
