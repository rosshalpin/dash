# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
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
    pie_fig = px.pie(data, values='percentage', names='sentiment', title='Normalised Sentiment Breakdown', hole=.5, width=500)
    pie_fig.update_layout(
        font_family="monospace"
    )
    return pie_fig


def get_word_cloud(text):
    fig = px.imshow(WordCloud(max_words=50,
        background_color="white",
        scale=3
        ).generate(text),title="Input Data Word Cloud")
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

# @app.callback(
#     Output('wordCloud', 'figure'),
#     [Input('pie-graph', 'clickData')])
# def display_click_data(clickData):

#     return get_word_cloud(" ".join(df.text.str.split(expand=True).stack()))

if __name__ == '__main__':
    app.run_server(debug=True)