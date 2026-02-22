import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import plotly.io as pio

try:
    from ssqueezepy import ssq_cwt
    from ssqueezepy.wavelets import Wavelet
    HAS_SSQ = True
except Exception:
    HAS_SSQ = False


def load_uploaded_npy(uploaded_file):
    try:
        arr = np.load(uploaded_file, allow_pickle=True)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(float)
        return None
    except Exception:
        return None


def scales_to_periods(scales, wavelet_name, fs):
    """
    Convert CWT scales -> periods using wavelet center frequency approximation.
    For morlet, this is usually fine for publication-grade visualization.
    """
    w = Wavelet(wavelet_name)
    wc = float(getattr(w, "wc", 2 * np.pi))  # fallback if version differs
    freqs = (wc * float(fs)) / (2 * np.pi * np.maximum(scales, 1e-12))
    periods = 1.0 / np.maximum(freqs, 1e-12)
    return periods


def interp_cwt_to_sswt_period_grid(mag_cwt, periods_cwt, periods_target):
    """
    Interpolate |CWT| along period axis onto SSWT period grid for each time column.
    Interp done in log10(period) domain to respect log axis.
    """
    periods_cwt = np.asarray(periods_cwt, dtype=float)
    periods_target = np.asarray(periods_target, dtype=float)

    # Ensure finite positive periods
    m1 = np.isfinite(periods_cwt) & (periods_cwt > 0)
    periods_cwt = periods_cwt[m1]
    mag_cwt = mag_cwt[m1, :]

    # Sort by period ascending for interpolation
    order = np.argsort(periods_cwt)
    periods_cwt = periods_cwt[order]
    mag_cwt = mag_cwt[order, :]

    log_p_cwt = np.log10(periods_cwt)
    log_p_tgt = np.log10(np.maximum(periods_target, 1e-12))

    out = np.empty((len(periods_target), mag_cwt.shape[1]), dtype=float)

    # Vectorized per-column interpolation (loop over time, but lightweight)
    for t in range(mag_cwt.shape[1]):
        col = mag_cwt[:, t]
        out[:, t] = np.interp(log_p_tgt, log_p_cwt, col, left=0.0, right=0.0)

    return out


def analyze_and_build_figure(
    data, fps,
    nv,
    y_min, y_max,
    ridge_thresh_percent,
    min_dist, top_k_ridges,
    transition_duration_sec,
    transition_ratio,
):
    # Force morlet for both, per your request
    wavelet = "morlet"

    # ---- 1) Compute SSQ-CWT once: returns Tx (SSWT), Wx (CWT), ssq_freqs, scales ----
    try:
        Tx, Wx, ssq_freqs, scales = ssq_cwt(data, wavelet=wavelet, fs=fps, nv=nv)
    except Exception as e:
        st.error(f"ssq_cwt 計算失敗：{e}")
        return None, [], {}

    mag_sswt = np.abs(Tx)
    mag_cwt = np.abs(Wx)

    with np.errstate(divide="ignore", invalid="ignore"):
        periods_sswt = 1.0 / ssq_freqs

    periods_cwt = scales_to_periods(scales, wavelet, fps)

    time_axis = np.arange(len(data)) / fps
    total_duration = float(time_axis[-1]) if len(time_axis) else 0.0

    # ---- 2) Clamp to display band using SSWT grid (target grid) ----
    valid_target = np.isfinite(periods_sswt) & (periods_sswt >= y_min) & (periods_sswt <= y_max)
    periods_plot = periods_sswt[valid_target]

    # ---- 3) Interpolate CWT magnitude onto SSWT period grid so (a) is truly "broader" counterpart ----
    mag_cwt_on_sswt = interp_cwt_to_sswt_period_grid(mag_cwt, periods_cwt, periods_sswt)
    mag_cwt_plot = mag_cwt_on_sswt[valid_target, :]

    # ---- 4) Ridge extraction on SSWT within band ----
    valid_sswt = valid_target
    band_mag = np.where(valid_sswt[:, None], mag_sswt, 0.0)
    global_max = float(np.max(band_mag)) if band_mag.size else 0.0
    abs_threshold = global_max * float(ridge_thresh_percent)

    harmonic_data = {1: {"x": [], "y": [], "z": []},
                     2: {"x": [], "y": [], "z": []},
                     3: {"x": [], "y": [], "z": []},
                     0: {"x": [], "y": [], "z": []}}

    transition_events = []
    consecutive = 0
    required_frames = int(max(0.0, transition_duration_sec) * fps)
    current_start = None
    in_transition = False

    nT = mag_sswt.shape[1]

    for t_idx in range(nT):
        slice_ = np.array(mag_sswt[:, t_idx], copy=True)
        slice_[~valid_sswt] = 0.0

        peaks, props = find_peaks(slice_, height=abs_threshold, distance=int(min_dist))

        if len(peaks) > 0:
            peak_periods = periods_sswt[peaks]
            peak_energies = props.get("peak_heights", np.array([]))

            order = np.argsort(peak_energies)[::-1]
            keep = order[:int(top_k_ridges)]
            final_periods = peak_periods[keep]
            final_energies = peak_energies[keep]

            base_idx = int(np.argmax(final_periods))  # longest period = fundamental
            T_base = float(final_periods[base_idx])
            E_base = float(final_energies[base_idx])
            t_val = float(time_axis[t_idx])

            # classify peaks
            for p_val, e_val in zip(final_periods, final_energies):
                p_val = float(p_val)
                e_val = float(e_val)

                if p_val <= 0 or not np.isfinite(p_val):
                    h = 0
                else:
                    ratio = T_base / p_val
                    if 0.85 <= ratio <= 1.15:
                        h = 1
                    elif 1.8 <= ratio <= 2.2:
                        h = 2
                    elif 2.8 <= ratio <= 3.2:
                        h = 3
                    else:
                        h = 0

                harmonic_data[h]["x"].append(t_val)
                harmonic_data[h]["y"].append(p_val)
                harmonic_data[h]["z"].append(e_val)

            # transition detection
            mask_2 = (periods_sswt >= T_base / 2.2) & (periods_sswt <= T_base / 1.8)
            mask_3 = (periods_sswt >= T_base / 3.2) & (periods_sswt <= T_base / 2.8)
            E2 = float(np.max(slice_[mask_2])) if np.any(mask_2) else 0.0
            E3 = float(np.max(slice_[mask_3])) if np.any(mask_3) else 0.0
            min_required = E_base * 0.05

            if (E3 > E2 * float(transition_ratio)) and (E3 > min_required):
                if not in_transition:
                    current_start = t_val
                    in_transition = True
                consecutive += 1
            else:
                if in_transition and consecutive >= required_frames and current_start is not None:
                    transition_events.append(current_start)
                in_transition = False
                consecutive = 0
        else:
            if in_transition and consecutive >= required_frames and current_start is not None:
                transition_events.append(current_start)
            in_transition = False
            consecutive = 0

    if in_transition and consecutive >= required_frames and current_start is not None:
        transition_events.append(current_start)

    # turn = lowest fundamental period (global min on 1st harmonic)
    stats = {"turn_t": None, "turn_p": None}
    if len(harmonic_data[1]["y"]) > 0:
        y = np.array(harmonic_data[1]["y"], dtype=float)
        x = np.array(harmonic_data[1]["x"], dtype=float)
        m = np.isfinite(y) & (y > 0) & np.isfinite(x)
        y, x = y[m], x[m]
        if y.size:
            idx = int(np.argmin(y))
            stats["turn_t"] = float(x[idx])
            stats["turn_p"] = float(y[idx])

    # ---- 5) Build two-column figure with sane typography ----
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) CWT (Morlet, from ssq_cwt Wx)", "(b) Ridge Extraction (SSWT-based)"),
        horizontal_spacing=0.18,
    )

    # Left: CWT heatmap (on SSWT period grid for apples-to-apples)
    fig.add_trace(
        go.Heatmap(
            z=mag_cwt_plot,
            x=time_axis,
            y=periods_plot,
            coloraxis="coloraxis",
            name="CWT",
        ),
        row=1, col=1
    )

    # Right: ridge scatter (hide its colorbar to prevent crowding)
    labels = {1: "1st", 2: "2nd", 3: "3rd", 0: "Others"}
    symbols = {1: "circle", 2: "diamond", 3: "cross", 0: "x"}

    all_z = []
    for k in harmonic_data:
        all_z.extend(harmonic_data[k]["z"])
    zmin, zmax = (float(min(all_z)), float(max(all_z))) if all_z else (0.0, 1.0)

    for k in [1, 2, 3, 0]:
        d = harmonic_data[k]
        if not d["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["x"], y=d["y"],
                mode="markers",
                marker=dict(
                    symbol=symbols[k],
                    size=6 if k == 1 else 5,
                    color=d["z"],
                    cmin=zmin,
                    cmax=zmax if zmax > zmin else (zmin + 1.0),
                    colorscale="Viridis",
                    showscale=False,   # ✅ 不要再擠一個色條出來
                    line=dict(width=0.5, color="black") if k == 1 else None,
                ),
                name=labels[k],
                hovertemplate=f"<b>{labels[k]}</b><br>Time: %{{x:.2f}} s<br>Period: %{{y:.4f}} s<extra></extra>",
            ),
            row=1, col=2
        )

    # Transition line on both panels
    if transition_events:
        t0 = float(transition_events[0])
        fig.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color="black", row=1, col=1)
        fig.add_vline(x=t0, line_width=1.5, line_dash="dash", line_color="black", row=1, col=2)
        fig.add_annotation(
            x=t0, y=y_max,
            xref="x2", yref="y2",
            text="Transition",
            showarrow=False,
            yshift=16,
            font=dict(family="Arial", size=11, color="black"),
        )

    # Turn marker on right panel
    if stats["turn_t"] is not None and stats["turn_p"] is not None:
        fig.add_trace(
            go.Scatter(
                x=[stats["turn_t"]], y=[stats["turn_p"]],
                mode="markers+text",
                text=["turn"],
                textposition="top left",
                marker=dict(symbol="circle-open", size=12, line=dict(width=2.5, color="crimson")),
                name="turn",
                hovertemplate="<b>turn</b><br>Time: %{x:.2f} s<br>Period: %{y:.4f} s<extra></extra>",
            ),
            row=1, col=2
        )

    # Layout: stop the text wars
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12, color="black"),
        margin=dict(t=110, b=70, l=80, r=90),
        height=520,
        showlegend=False,
        coloraxis=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="|CWT|",
                thickness=14,
                len=0.82,
                x=0.47,  # 放兩圖中間偏左
                y=0.5,
                outlinewidth=1,
                outlinecolor="black",
                tickfont=dict(size=10),
                titlefont=dict(size=11),
            ),
        ),
    )

    # Subplot titles: smaller + higher, avoid collisions
    fig.update_annotations(font=dict(size=14, family="Arial", color="black"), y=1.07)

    axis_settings = dict(
        showline=True, linecolor="black", linewidth=1.5,
        mirror=True, ticks="inside", tickcolor="black", tickwidth=1.5, ticklen=6,
        showgrid=False, zeroline=False,
    )

    y_range = [np.log10(y_min), np.log10(y_max)]

    fig.update_xaxes(title_text="Time (s)", title_standoff=10, range=[0, total_duration], **axis_settings, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", title_standoff=10, range=[0, total_duration], **axis_settings, row=1, col=2)
    fig.update_yaxes(title_text="Period (s)", title_standoff=12, type="log", range=y_range, **axis_settings, row=1, col=1)
    fig.update_yaxes(title_text="Period (s)", title_standoff=12, type="log", range=y_range, **axis_settings, row=1, col=2)

    return fig, transition_events, stats


# ================= Streamlit App =================
st.set_page_config(page_title="Two-column: CWT + Ridge", layout="wide")
st.title("Two-column Figure: (a) CWT (original Wx) + (b) Ridge Extraction")

if not HAS_SSQ:
    st.error("缺套件：pip install -U ssqueezepy scipy plotly streamlit numpy kaleido")
    st.stop()

with st.sidebar:
    st.header("Parameters")
    fps = st.number_input("Sampling Rate (FPS)", value=30.0, min_value=1.0)
    nv = st.select_slider("SSWT Voices (nv)", options=[16, 32, 64], value=32)

    st.subheader("Period range (s)")
    c1, c2 = st.columns(2)
    y_min = c1.number_input("Min Period (s)", value=0.1, min_value=1e-6, format="%.6f")
    y_max = c2.number_input("Max Period (s)", value=10.0, min_value=1e-6, format="%.6f")

    st.subheader("Ridge extraction")
    ridge_thresh = st.slider("Energy Threshold (%)", 1, 40, 5)
    min_dist = st.slider("Min Peak Distance (px)", 1, 50, 15)
    top_k = st.slider("Keep Top K Peaks", 1, 10, 5)

    st.subheader("Transition")
    transition_dur = st.number_input("Trigger Duration (s)", value=0.1, step=0.05, min_value=0.0)
    transition_ratio = st.slider("E3 > E2 multiplier", 1.0, 3.0, 1.0, 0.1)

    st.subheader("PDF export")
    export_width = st.number_input("PDF width (px)", value=1700, min_value=900, step=100)
    export_height = st.number_input("PDF height (px)", value=520, min_value=400, step=20)

uploaded_file = st.file_uploader("Upload .npy Data File", type=["npy"])

if uploaded_file is not None:
    x = load_uploaded_npy(uploaded_file)
    if x is None:
        st.error("讀檔失敗：需要 1D array 或 2D array（至少兩欄，取第 2 欄）。")
        st.stop()

    x = x - np.mean(x)

    fig, transitions, stats = analyze_and_build_figure(
        data=x, fps=fps,
        nv=nv,
        y_min=y_min, y_max=y_max,
        ridge_thresh_percent=ridge_thresh / 100.0,
        min_dist=min_dist,
        top_k_ridges=top_k,
        transition_duration_sec=transition_dur,
        transition_ratio=transition_ratio,
    )

    if fig is None:
        st.stop()

    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### Download PDF")
    try:
        fig_export = go.Figure(fig.to_dict())
        fig_export.update_layout(width=int(export_width), height=int(export_height))
        pdf_bytes = pio.to_image(fig_export, format="pdf", engine="kaleido")
        st.download_button(
            "Download two-column figure (PDF)",
            data=pdf_bytes,
            file_name="two_column_CWTWx_ridge.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error("PDF 匯出失敗，通常是 kaleido 沒裝：pip install -U kaleido\n\n" + str(e))

    st.markdown("### Summary")
    if stats.get("turn_t") is not None:
        st.write(f"- turn: {stats['turn_t']:.2f} s, period = {stats['turn_p']:.4f} s (~ {1/stats['turn_p']:.2f} Hz)")
    else:
        st.write("- turn: not detected")
    if transitions:
        st.write(f"- Transition (first): {transitions[0]:.2f} s, total = {len(transitions)}")
    else:
        st.write("- Transition: none detected")
