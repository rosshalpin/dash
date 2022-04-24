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
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
import plotly.figure_factory as ff

app = Dash(__name__)


path = os.path.abspath(os.getcwd())
data_path=f"{path}/static/data/"


df = pd.read_csv(data_path+"cleaned_tweets.csv")

def load_pickled_data(file_name):
    with open(f'{data_path}{file_name}.pickle', 'rb') as handle:
        return pickle.load(handle)


svm_lc = load_pickled_data('svm_lc')
dnn = load_pickled_data('dnn')
svm = load_pickled_data('svm')
dnn_history = load_pickled_data('history')

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
                                 'positive':'limegreen',
                                 'neutral':'blue'}
    )
    pie_fig.update_layout(
        font_family="monospace"
    )
    return pie_fig


def get_word_cloud(text, title="mixed"):
    color_map = {
        "negative": lambda *args, **kwargs: (231,15,15),
        "neutral": lambda *args, **kwargs: (35,76,255),
        "positive": lambda *args, **kwargs: (51,186,15),
        "mixed": None
    }
    fig = px.imshow(WordCloud(max_words=50,
        background_color="white",
        color_func=color_map[title],
        scale=3,
        ).generate(text),

        title=f"Top 50 Word Cloud for sentiment: {title}<br>(click to reset)",
        binary_compression_level=0)
    fig.update_layout(
        showlegend=False,
        font_family="monospace"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


def get_geo_plot(df):
    coords = df.coords.str.split(expand=True, pat=',')
    df_ = df.copy(deep=True)
    df_['lat'] = coords[0]
    df_['lon'] = coords[1]
    fig = px.scatter_geo(
            df_, lat='lat', lon='lon',
            color="sentiment", # which column to use to set the color of markers
            hover_name="place", # column added to hover information
            projection="natural earth",
            title='Scatter Geo Plot of some tweet sentiments',
            color_discrete_map={'negative':'orangered',
                                 'positive':'limegreen',
                                 'neutral':'blue'},
        )
    fig.update_layout(
        font_family="monospace"
    )
    return fig


def get_confusion_matrix(data):
    y_test, y_pred = data
    cm = confusion_matrix(y_test, y_pred)
    x = ["negative", "neutral", "positive"]
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(cm, x=x, y=x, annotation_text=z_text, colorscale='Viridis')
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=1.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.21,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    fig.update_layout(
        margin=dict(t=100, l=100),
        font_family="monospace",
        title="SVM Confusion Matrix"
    )
    return fig


def get_svm_lc(data):
    train_sizes, train_scores, test_scores, fit_times = data
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_df = pd.DataFrame(zip(train_scores_mean, train_sizes), columns=["scores", "sizes"])
    train_df['type'] = "train"
    test_df = pd.DataFrame(zip(test_scores_mean, train_sizes), columns=["scores", "sizes"])
    test_df['type'] = "validation"
    lc_df = pd.concat([train_df,test_df])
    fig = px.line(lc_df, x="sizes", y="scores",color='type', title='SVM Learning Curve Accuracy')
    fig.update_layout(
        font_family="monospace"
    )
    return fig

app.layout = html.Div(children=[
    html.H1(children='Twitter data dashboard'),

    html.Div(children='''
        Data Visualisation (DV) Common Module Assessment (CMA)
    '''),

    html.Div(children='''
    '''),

    html.Br(),
    html.H3(children='Input Data'),
    html.Br(),

    html.Div([
        dcc.Graph(
            id='pie-graph',
            figure=get_pie_chart(df),
            style={'width': '33vw',"display": "inline-block"}
        ),

        dcc.Graph(
            id="wordCloud",
            figure=get_word_cloud(" ".join(df.text.str.split(expand=True).stack())),
            style={'width': '33vw', "display": "inline-block"}
        ),
        dcc.Graph(
            id="geoPlot",
            figure=get_geo_plot(df),
            style={'width': '33vw', "display": "inline-block"}
        ),
    ]),
    html.Br(),
    html.H3(children='SVM Performance'),
    html.Br(),
    html.Div([
        dcc.Graph(
            id='svm-lc',
            figure=get_svm_lc(svm_lc),
            style={'width': '33vw',"display": "inline-block"}
        ),
        dcc.Graph(
            id='svm_cm',
            figure=get_confusion_matrix(svm),
            style={'width': '33vw',"display": "inline-block"}
        ),
    ]),
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
