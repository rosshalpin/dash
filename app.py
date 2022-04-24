# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import dash
# import dash_bootstrap_components as dbc
import plotly.express as px
import os
import pandas as pd
from wordcloud import WordCloud
from dash.dependencies import Input, Output
import json

app = Dash(__name__)


path = os.path.abspath(os.getcwd())
data_path=f"{path}/static/data/"


df = pd.read_csv(data_path+"cleaned_tweets.csv")

def get_pie_chart(df):
    sent_counts = df.sentiment.value_counts(normalize=True) * 100
    data = pd.DataFrame(zip(sent_counts.keys(),sent_counts.values), columns=["sentiment", "percentage"])
    pie_fig = px.pie(data, 
        values='percentage', 
        names='sentiment', 
        title='Normalised Sentiment Breakdown<br>(click to obtain Sentiment Word Cloud)', 
        hole=.5, 
        width=500,
        color='sentiment',
        color_discrete_map={'negative':'orangered',
                                 'positive':'lightgreen',
                                 'neutral':'lightgrey'}
    )
    pie_fig.update_layout(
        font_family="monospace"
    )
    return pie_fig


def get_word_cloud(text, title="mixed"):
    color_map = {
        "negative": lambda *args, **kwargs: (231,15,15),
        "neutral": lambda *args, **kwargs: (125,125,125),
        "positive": lambda *args, **kwargs: (51,186,15),
        "mixed": None
    }
    fig = px.imshow(WordCloud(max_words=50,
        background_color="white",
        color_func=color_map[title],
        scale=3
        ).generate(text),title=f"Top 50 Word Cloud for sentiment: {title}<br>(click to reset)")
    fig.update_layout(
        showlegend=False,
        font_family="monospace"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


app.layout = html.Div(children=[
    html.H1(children='Twitter data dashboard'),

    html.Div(children='''
        Data Visualisation (DV) Common Module Assessment (CMA)
    '''),

    html.Div(children='''
    '''),

    dcc.Graph(
        id='pie-graph',
        figure=get_pie_chart(df),
        style={'width': '10md', 'height': '30md',"display": "inline-block","marginLeft": "10"}
    ),

    dcc.Graph(
        id="wordCloud",
        figure=get_word_cloud(" ".join(df.text.str.split(expand=True).stack())),
        style={"display": "inline-block","marginLeft": "10"}
    ),
    html.Div([
        dcc.Markdown(("""
            **Click Data**

            Click on points in the graph.
        """)),
        html.Pre(id='click-data'),
    ], className='three columns'), 
])

@app.callback(
    Output('click-data', 'children'),
    [Input('pie-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('wordCloud', 'figure'),
    [
        Input('pie-graph', 'clickData'),
        Input('wordCloud', 'clickData')
    ],
    prevent_initial_call=True)
def update_pie_graph(clickData, points):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    if "pie" in trigger:
        sentiment = clickData["points"][0]["label"]
        df_sent = df.query(f'sentiment == "{sentiment}"')
        words = " ".join(df_sent.text.str.split(expand=True).stack())
        return get_word_cloud(words, title=sentiment)
    if "word" in trigger:
        return get_word_cloud(" ".join(df.text.str.split(expand=True).stack()))


if __name__ == '__main__':
    app.run_server(debug=True)