import plotly.graph_objects as go

categories = ['Qwen2-0.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Mistral-7B-Instruct-v0.2', 'Llama3.1-8B-Instruct']
group1 = [0.4812, 0.7091, 0.8197, 0.85]
group2 = [0.2963, 0.7797, 0.413, 0.8148]
group3 = [0.2889, 0.6724, 0.4359, 0.6098]

fig = go.Figure(data=[
    go.Bar(name='Llama-3.3-70B-Instruct', x=categories, y=group1),
    go.Bar(name='Qwen2.5-75B-Instruct', x=categories, y=group2),
    go.Bar(name='Mistral-7B-Instruct-v0.2', x=categories, y=group3)
])

fig.update_layout(
    barmode='group',
    title='ASR when attacking GPT-4.1 by converting its stronger answers from SAE to AAE',
    xaxis_title='Weaker Models, whose answers are not converted',
    yaxis_title='ASR (Attack Success Rate)',
    legend=dict(
        title='Judge Models'
    )
)


fig.show()