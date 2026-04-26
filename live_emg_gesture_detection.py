import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Load saved model files ────────────────────────────────────
with open("emg_gesture_model.pkl","rb") as f: model       = pickle.load(f)
with open("emg_scaler.pkl","rb")        as f: scaler      = pickle.load(f)
with open("emg_gesture_map.pkl","rb")   as f: gesture_map = pickle.load(f)
with open("emg_columns.pkl","rb")       as f: COLS        = pickle.load(f)

COLORS = {
    'HandOpen' : '#2ecc71',
    'HandClose': '#5dade2',
    'Rest'     : '#a855f7'
}

TEST_FILES = {
    'HandOpen' : r"test_files\tableConvert.com_taxyfp.csv",
    'HandClose': r"test_files\tableConvert.com_ab4l5w.csv",
    'Rest'     : r"test_files\tableConvert.com_p4f4v9.csv",
}

CHUNK_SIZE = 500
REST_SIZE  = 500

def load_single_file(filepath, gesture_name):
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    values = df.iloc[:, 0].values.astype(float)
    result = pd.DataFrame(0.0, index=range(len(values)), columns=COLS)
    if gesture_name in result.columns:
        result[gesture_name] = values
    else:
        result[COLS[0]] = values
    result['true_gesture'] = gesture_name
    result['signal_raw']   = values
    return result

def build_mixed_signal(test_files, chunk_size, rest_size):
    import random
    random.seed(42)

    data = {}
    for gesture_name, filepath in test_files.items():
        df = load_single_file(filepath, gesture_name)
        data[gesture_name] = df
        print(f"Loaded: {gesture_name} | Samples: {len(df)}")

    rest_df  = data['Rest']
    # More HandClose by weighting it higher
    gestures = ['HandOpen','HandClose','HandClose','HandClose']
    pointers = {g: 0 for g in set(gestures)}
    rest_ptr = 0
    segments = []

    for _ in range(20):
        g     = random.choice(gestures)
        ptr   = pointers[g]
        chunk = data[g].iloc[ptr: ptr + chunk_size].copy()
        if len(chunk) == 0:
            pointers[g] = 0
            chunk = data[g].iloc[0: chunk_size].copy()
        pointers[g] = (ptr + chunk_size) % len(data[g])
        segments.append(chunk)

        rptr       = rest_ptr
        rest_chunk = rest_df.iloc[rptr: rptr + rest_size].copy()
        rest_chunk['true_gesture'] = 'Rest'
        if len(rest_chunk) < rest_size:
            rest_ptr  = 0
            rest_chunk = rest_df.iloc[0: rest_size].copy()
            rest_chunk['true_gesture'] = 'Rest'
        rest_ptr = (rptr + rest_size) % len(rest_df)
        segments.append(rest_chunk)

    return pd.concat(segments, ignore_index=True)

def predict_chunk(rows):
    X_sc  = scaler.transform(rows)
    preds = model.predict(X_sc)
    unique, counts = np.unique(preds, return_counts=True)
    majority = unique[np.argmax(counts)]
    return majority, gesture_map.get(majority, str(majority))

def get_ref_wave(filepath, n=300):
    """Get most active N-sample segment as reference waveform."""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    vals = df.iloc[:, 0].values.astype(float)
    best_s, best_e = 0, 0
    for s in range(0, len(vals) - n, n):
        e = np.sum(vals[s:s+n])
        if e > best_e:
            best_e = e; best_s = s
    return vals[best_s: best_s + n]

def draw_ref_axes(axes_ref, ref_waves, Y_MAX):
    """Draw reference waveforms in top 3 axes."""
    for ax, (gname, wave) in zip(axes_ref, ref_waves.items()):
        ax.cla()
        ax.set_facecolor('#0a0f1a')
        ax.tick_params(colors='#555', labelsize=6)
        ax.spines['bottom'].set_color('#222')
        ax.spines['left'].set_color('#222')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, Y_MAX)
        ax.set_xlim(0, len(wave))
        ax.set_yticks([])
        ax.set_xticks([])

        c = COLORS[gname]
        x = np.arange(len(wave))

        ax.vlines(x, 0, wave, color=c, linewidth=0.6, alpha=0.85)
        ax.fill_between(x, 0, wave, color=c, alpha=0.08)

        # Envelope
        env = pd.Series(wave).rolling(10, center=True, min_periods=1).max().values
        ax.plot(x, env, color=c, linewidth=1.0, alpha=0.5, linestyle='--')

        # Label
        ax.text(0.5, 0.92, gname,
                transform=ax.transAxes,
                fontsize=8, fontweight='bold',
                color=c, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='#0a0f1a',
                          edgecolor=c, linewidth=1.2,
                          alpha=0.9))

        ax.text(0.5, 0.08, "REFERENCE",
                transform=ax.transAxes,
                fontsize=6, color='#555',
                ha='center', va='bottom',
                family='monospace')

def run_realtime(test_files, chunk_size=CHUNK_SIZE,
                 rest_size=REST_SIZE, delay=0.001):

    # ── Load reference waveforms ──────────────────────────────
    ref_waves = {}
    for gname, fpath in test_files.items():
        ref_waves[gname] = get_ref_wave(fpath, n=1000)
    print("Reference waveforms loaded.")

    # ── Build mixed signal ────────────────────────────────────
    df_all        = build_mixed_signal(test_files, chunk_size, rest_size)
    total         = len(df_all)
    true_gestures = df_all['true_gesture'].tolist()
    signal_arr    = df_all['signal_raw'].values.astype(float)
    Y_MAX         = signal_arr.max() * 1.12

    # ── Pre-compute chunk predictions ────────────────────────
    print("\nPre-computing predictions...")
    chunk_boundaries = []
    chunk_pred_label = []

    i = 0
    while i < total:
        true_now = true_gestures[i]
        j = i
        while j < total and true_gestures[j] == true_now:
            j += 1
        _, pred = predict_chunk(df_all[COLS].iloc[i:j].values)
        chunk_boundaries.append((i, j, true_now, pred))
        chunk_pred_label.extend([pred] * (j - i))
        print(f"  [{i:5d}:{j:5d}]  True: {true_now:<12} "
              f"Pred: {pred:<12} "
              f"{'✓' if pred==true_now else '✗'}")
        i = j

    chunk_pred_arr = np.array(chunk_pred_label)
    print(f"\nReady — {total} samples, {len(chunk_boundaries)} chunks\n")

    # ── Figure layout ─────────────────────────────────────────
    # Row 0 (small): 3 reference waveforms side by side
    # Row 1 (large): live signal
    plt.ion()
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1, 3],
        hspace=0.35,
        wspace=0.06,
        top=0.93, bottom=0.09,
        left=0.06, right=0.97
    )

    axes_ref  = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax_live   = fig.add_subplot(gs[1, :])   # spans all 3 columns

    # Style live axis once
    ax_live.set_facecolor('#0d1117')

    UPDATE = 15

    for frame in range(1, total, UPDATE):

        # ── Draw reference waveforms (top) ────────────────────
        draw_ref_axes(axes_ref, ref_waves, Y_MAX)

        # Highlight the currently detected gesture's ref panel
        pred_now  = chunk_pred_arr[frame]
        true_now  = true_gestures[frame]
        correct   = pred_now == true_now

        for ax_r, gname in zip(axes_ref, ref_waves.keys()):
            is_active = (gname == pred_now)
            lw   = 2.5  if is_active else 0.8
            alpha= 1.0  if is_active else 0.3
            col  = COLORS[gname]
            for spine in ax_r.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(lw)
                spine.set_alpha(alpha)
            # Glowing top label when active
            if is_active:
                ax_r.set_title(
                    "▶ DETECTED",
                    fontsize=8, fontweight='bold',
                    color=col, pad=3)
            else:
                ax_r.set_title("", pad=3)

        # ── Live signal (bottom) ──────────────────────────────
        ax_live.cla()
        ax_live.set_facecolor('#0d1117')
        ax_live.tick_params(colors='#8b949e', labelsize=9)
        ax_live.spines['bottom'].set_color('#30363d')
        ax_live.spines['left'].set_color('#30363d')
        ax_live.spines['top'].set_visible(False)
        ax_live.spines['right'].set_visible(False)
        ax_live.grid(True, alpha=0.1, color='#8b949e',
                     linewidth=0.4, linestyle='--', axis='y')

        ax_live.set_xlim(0, frame + 80)
        ax_live.set_ylim(0, Y_MAX)
        ax_live.set_ylabel("Signal Amplitude (mV)",
                           fontsize=10, color='#8b949e')
        ax_live.set_xlabel("Time (samples)",
                           fontsize=10, color='#8b949e')

        x_now = np.arange(0, frame + 1)
        s_now = signal_arr[0: frame + 1]
        p_now = chunk_pred_arr[0: frame + 1]

        # Draw spike waveform colored by predicted gesture
        seg_start = 0
        seg_label = p_now[0]
        for k in range(1, len(x_now)):
            if p_now[k] != seg_label or k == len(x_now) - 1:
                sx = x_now[seg_start: k+1]
                sy = s_now[seg_start: k+1]
                c  = COLORS.get(seg_label, '#fff')
                ax_live.vlines(sx, 0, sy,
                               color=c, linewidth=0.6, alpha=0.85)
                ax_live.fill_between(sx, 0, sy,
                                     color=c, alpha=0.07)
                seg_start = k
                seg_label = p_now[k]

        # Background shading + chunk labels
        for (cs, ce, ct, cp) in chunk_boundaries:
            xe = min(ce, frame + 1)
            if cs >= xe:
                break
            ax_live.axvspan(cs, xe,
                            color=COLORS.get(cp, '#333'),
                            alpha=0.06, zorder=0)
            mid = (cs + min(ce, frame)) // 2
            if cs < frame:
                ax_live.text(mid, Y_MAX * 0.97, cp,
                             fontsize=7, color=COLORS.get(cp,'#fff'),
                             ha='center', va='top',
                             fontweight='bold', alpha=0.65)

        # Current position marker
        ax_live.axvline(x=frame, color='white',
                        linewidth=0.8, linestyle=':', alpha=0.4)

        # DETECTED pill
        box_color = COLORS.get(pred_now, '#fff')
        ax_live.text(0.01, 0.93,
                     f"▶  {pred_now}",
                     transform=ax_live.transAxes,
                     fontsize=13, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.45',
                               facecolor=box_color, alpha=0.90,
                               edgecolor='white', linewidth=1.2),
                     zorder=6)

        ax_live.text(0.01, 0.76,
                     f"true: {true_now}",
                     transform=ax_live.transAxes,
                     fontsize=9, color='#8b949e',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#161b22', alpha=0.85,
                               edgecolor='#30363d'),
                     zorder=6)

        ax_live.text(0.99, 0.93,
                     "✓ CORRECT" if correct else "✗ WRONG",
                     transform=ax_live.transAxes,
                     fontsize=10, fontweight='bold', ha='right',
                     color='#2ecc71' if correct else '#e74c3c',
                     bbox=dict(boxstyle='round,pad=0.35',
                               facecolor='#161b22', alpha=0.85,
                               edgecolor='#30363d'),
                     zorder=6)

        done = [(cs,ce,ct,cp) for cs,ce,ct,cp
                in chunk_boundaries if ce <= frame+1]
        acc  = (sum(1 for _,__,ct,cp in done if cp==ct)
                / len(done) * 100) if done else 0.0

        ax_live.text(0.99, 0.76,
                     f"Acc: {acc:.1f}%",
                     transform=ax_live.transAxes,
                     fontsize=9, ha='right', color='#8b949e',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#161b22', alpha=0.85,
                               edgecolor='#30363d'),
                     zorder=6)

        patches = [mpatches.Patch(color=COLORS[g], label=g, alpha=0.85)
                   for g in COLORS]
        ax_live.legend(handles=patches, loc='upper center', ncol=3,
                       fontsize=9, framealpha=0.25,
                       facecolor='#161b22', edgecolor='#30363d',
                       labelcolor='white',
                       bbox_to_anchor=(0.5, 1.02))

        fig.suptitle(
            f"EMG Gesture Recognition  ·  "
            f"Sample {frame:,} / {total:,}  ·  "
            f"Chunk Acc: {acc:.1f}%",
            fontsize=12, fontweight='bold',
            color='white', y=0.99)

        plt.pause(delay)

    plt.ioff()

    # ── Final result ──────────────────────────────────────────
    correct_c = sum(1 for _,__,ct,cp in chunk_boundaries if cp==ct)
    total_c   = len(chunk_boundaries)
    final_acc = correct_c / total_c * 100
    print(f"\n{'='*55}")
    print(f"  FINAL CHUNK ACCURACY : {final_acc:.2f}%")
    print(f"  Correct : {correct_c} / {total_c} chunks")
    print(f"{'='*55}")

    # ── Final static full-signal plot ─────────────────────────
    fig2 = plt.figure(figsize=(18, 9))
    fig2.patch.set_facecolor('#0d1117')
    gs2 = gridspec.GridSpec(2, 3, figure=fig2,
                             height_ratios=[1, 3], hspace=0.35,
                             wspace=0.06, top=0.93, bottom=0.09,
                             left=0.06, right=0.97)

    axes_ref2 = [fig2.add_subplot(gs2[0, k]) for k in range(3)]
    ax2       = fig2.add_subplot(gs2[1, :])

    draw_ref_axes(axes_ref2, ref_waves, Y_MAX)
    for ax_r in axes_ref2:
        for spine in ax_r.spines.values():
            spine.set_linewidth(0.8)

    ax2.set_facecolor('#0d1117')
    ax2.tick_params(colors='#8b949e', labelsize=9)
    ax2.spines['bottom'].set_color('#30363d')
    ax2.spines['left'].set_color('#30363d')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.1, color='#8b949e',
             linewidth=0.4, linestyle='--', axis='y')
    ax2.set_xlim(0, total)
    ax2.set_ylim(0, Y_MAX)
    ax2.set_ylabel("Signal Amplitude (mV)", fontsize=10, color='#8b949e')
    ax2.set_xlabel("Time (samples)",        fontsize=10, color='#8b949e')

    seg_start = 0
    seg_label = chunk_pred_label[0]
    for k in range(1, total):
        if chunk_pred_label[k] != seg_label or k == total - 1:
            sx = np.arange(seg_start, k+1)
            sy = signal_arr[seg_start: k+1]
            c  = COLORS.get(seg_label, '#fff')
            ax2.vlines(sx, 0, sy, color=c, linewidth=0.4, alpha=0.85)
            ax2.fill_between(sx, 0, sy, color=c, alpha=0.07)
            seg_start = k
            seg_label = chunk_pred_label[k]

    for (cs, ce, ct, cp) in chunk_boundaries:
        ax2.axvspan(cs, ce, color=COLORS.get(cp,'#333'), alpha=0.06, zorder=0)
        ax2.text((cs+ce)//2, Y_MAX*0.97, cp, fontsize=6,
                 color=COLORS.get(cp,'#fff'), ha='center', va='top',
                 fontweight='bold', alpha=0.65)

    patches = [mpatches.Patch(color=COLORS[g], label=g, alpha=0.85)
               for g in COLORS]
    ax2.legend(handles=patches, loc='upper center', ncol=3, fontsize=9,
               framealpha=0.25, facecolor='#161b22', edgecolor='#30363d',
               labelcolor='white', bbox_to_anchor=(0.5, 1.02))

    fig2.suptitle(
        f"Full EMG Signal  ·  "
        f"Chunk Accuracy: {final_acc:.2f}%  ·  "
        f"{correct_c}/{total_c} chunks correct",
        fontsize=12, fontweight='bold', color='white', y=0.99)

    plt.show()

# ── Run ───────────────────────────────────────────────────────
run_realtime(TEST_FILES, chunk_size=500, rest_size=150, delay=0.001)