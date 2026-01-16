import math

import plotly.graph_objects as go

categories = ['Mistral-7B-Instruct-v0.2', 'Qwen2.5-72B-Instruct', 'Qwen3-32B', 'phi-4', 'Llama-3.3-70B-Instruct']
# keep original trace names in same order as data arrays
trace_info = [
    ('Llama-2-7b-chat-hf', [0.23, 0.20, 0.74, 0.57, 0.0]),
    ('Llama3.1-8B',           [0.61, 0.81, 0.83, 0.78, 0.0]),
    ('Mistral-7B-Instruct-v0.2',[0.44, 0.41, 0.0, 0.77, 0.82]),
    ('Qwen2-0.5B-Instruct',   [0.29, 0.30, 0.0, 0.22, 0.48]),
    ('Qwen2.5-3B-Instruct',   [0.67, 0.0, 0.0, 0.69, 0.71]),
    ('gemini-1.5-flash',      [0.75, 0.95, 0.94, 0.98, 0.98])
]

# optional color list
colors = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3']

# Convert to numpy arrays for easier indexing
n_categories = len(categories)
n_traces = len(trace_info)

# Build a list of bars (each bar has x numeric, y value, text, trace name, color)
bars = []  # list of (trace_idx, cat_idx, x_pos, y, text)
for trace_idx, (_, vals) in enumerate(trace_info):
    for cat_idx, v in enumerate(vals):
        # treat zero as missing
        if v is None or (isinstance(v, float) and math.isnan(v)) or v == 0:
            continue
        bars.append((trace_idx, cat_idx, v))

# For each category compute how many bars are present and their offsets
# We'll use a small width per bar and compute centered offsets
bar_width = 0.12  # adjust for spacing; smaller -> more packed
# collect per-category lists of (trace_idx, y)
per_cat = {i: [] for i in range(n_categories)}
for trace_idx, cat_idx, v in bars:
    per_cat[cat_idx].append((trace_idx, v))

traces_to_plot = []  # to hold data per original trace (we will combine per-trace multiple segments)
# We'll create one go.Bar per original trace, but its x and y will be lists of positions for categories where it exists
trace_x = [[] for _ in range(n_traces)]
trace_y = [[] for _ in range(n_traces)]
trace_text = [[] for _ in range(n_traces)]

for cat_idx in range(n_categories):
    present = per_cat[cat_idx]  # list of (trace_idx,y)
    k = len(present)
    if k == 0:
        continue
    # centered offsets: e.g., for k=3 -> offsets = [-w, 0, +w], for k=2 -> [-w/2, +w/2]
    if k == 1:
        offsets = [0.0]
    else:
        step = bar_width
        start = - ( (k-1) * step ) / 2.0
        offsets = [start + i*step for i in range(k)]
    # assign offsets to present bars in a stable ordering (by trace_idx for consistent positions)
    present_sorted = sorted(present, key=lambda t: t[0])
    for offs, (trace_idx, y) in zip(offsets, present_sorted):
        x_pos = cat_idx + offs
        trace_x[trace_idx].append(x_pos)
        trace_y[trace_idx].append(y)
        trace_text[trace_idx].append(f"{y:.2f}")

# Build figure: one go.Bar per original trace, but only categories where it had data
fig = go.Figure()
for i, (name, _) in enumerate(trace_info):
    if not trace_x[i]:
        continue
    fig.add_trace(go.Bar(
        name=name,
        x=trace_x[i],
        y=trace_y[i],
        text=trace_text[i],
        textposition='outside',
        marker=dict(color=colors[i % len(colors)], line=dict(width=0.5, color='white')),
        width=bar_width * 0.9
    ))

# set ticks to category centers and labels
fig.update_xaxes(tickmode='array',
                 tickvals=list(range(n_categories)),
                 ticktext=categories
)

fig.update_yaxes(range=[0, 1.1])

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

fig.update_layout(
    font=dict(
        size=20,
        color="black"
    ),
    xaxis_title=dict(
            text="Judge Models",
        ),
    yaxis_title=dict(
            text="ASR (Attack Success Rate)",
        ),
    barmode='overlay',
    bargap=0.15,
    plot_bgcolor='rgba(250,250,250,1)',
    height=520,
    legend=dict(
        title=dict(text="Weak answer by"),
        font=dict(size=20)
    )
)

fig.write_image("asr.svg", width=1600, height=400)