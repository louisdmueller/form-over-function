import plotly.graph_objects as go

categories = ['Llama3.1-8B-Instruct', 'Qwen2-0.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2-7B-Instruct']
group1 = [0.85, None, 0.7091, None]
group2 = [0.8148, None, 0.7797, None]
group3 = [0.6098, 0.2889, 0.6724, 0.6176]

fig = go.Figure(data=[
    go.Bar(name='Llama-3.3-70B-Instruct', x=categories, y=group1),
    go.Bar(name='Qwen2.5-75B-Instruct', x=categories, y=group2),
    go.Bar(name='Mistral-7B-Instruct-v0.2', x=categories, y=group3)
])

fig.update_layout(
    barmode='group',
    title='ASR when attacking GPT-4.1'
    xaxis_title='Weaker Models',
    yaxis_title='ASR (Attack Success Rate)',
    legend=dict(
        title='Judge Models'
    )
)


fig.show()