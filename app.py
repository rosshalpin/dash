# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
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


external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)


path = os.path.abspath(os.getcwd())
data_path = f"{path}/static/data/"


df = pd.read_csv(data_path + "cleaned_tweets.csv")


def load_pickled_data(file_name):
    with open(f"{data_path}{file_name}.pickle", "rb") as handle:
        return pickle.load(handle)


svm_lc = load_pickled_data("svm_lc")
dnn = load_pickled_data("dnn")
svm = load_pickled_data("svm")
dnn_history = load_pickled_data("history")


def get_pie_chart(sents):
    sent_counts = sents.value_counts(normalize=True) * 100
    data = pd.DataFrame(
        zip(sent_counts.keys(), sent_counts.values), columns=["sentiment", "percentage"]
    )
    pie_fig = px.pie(
        data,
        values="percentage",
        names="sentiment",
        title="Normalised Sentiment Breakdown<br>(click to obtain Sentiment Word Cloud)",
        hole=0.5,
        color="sentiment",
        color_discrete_map={
            "negative": "orangered",
            "positive": "limegreen",
            "neutral": "blue",
        },
    )
    pie_fig.update_layout(font_family="monospace")
    return pie_fig


def get_word_cloud(text, title="mixed"):
    color_map = {
        "negative": lambda *args, **kwargs: (231, 15, 15),
        "neutral": lambda *args, **kwargs: (35, 76, 255),
        "positive": lambda *args, **kwargs: (51, 186, 15),
        "mixed": None,
    }
    fig = px.imshow(
        WordCloud(
            max_words=50,
            background_color="white",
            color_func=color_map[title],
            scale=3,
        ).generate(text),
        title=f"Top 50 Word Cloud for sentiment: {title}<br>(click to reset)",
        binary_compression_level=0,
    )
    fig.update_layout(showlegend=False, font_family="monospace")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


def get_geo_plot(df):
    coords = df.coords.str.split(expand=True, pat=",")
    df_ = df.copy(deep=True)
    df_["lat"] = coords[0]
    df_["lon"] = coords[1]
    fig = px.scatter_geo(
        df_,
        lat="lat",
        lon="lon",
        color="sentiment",  # which column to use to set the color of markers
        hover_name="place",  # column added to hover information
        title="Scatter Geo Plot of some tweet sentiments",
        color_discrete_map={
            "negative": "orangered",
            "positive": "limegreen",
            "neutral": "blue",
        },
        width=800,
    )
    fig.update_layout(font_family="monospace")
    return fig


def get_confusion_matrix(data, title):
    y_test, y_pred = data
    cm = confusion_matrix(y_test, y_pred)
    x = ["negative", "neutral", "positive"]
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(
        cm, x=x, y=x, annotation_text=z_text, colorscale="Viridis"
    )
    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=1.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.21,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )
    fig.update_layout(margin=dict(t=100, l=100), font_family="monospace", title=title)
    return fig


def get_svm_lc(data):
    train_sizes, train_scores, test_scores, fit_times = data
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_df = pd.DataFrame(
        zip(train_scores_mean, train_sizes), columns=["scores", "sizes"]
    )
    train_df["type"] = "train"
    test_df = pd.DataFrame(
        zip(test_scores_mean, train_sizes), columns=["scores", "sizes"]
    )
    test_df["type"] = "validation"
    lc_df = pd.concat([train_df, test_df])
    fig = px.line(
        lc_df, x="sizes", y="scores", color="type", title="SVM Learning Curve Accuracy"
    )
    fig.update_layout(font_family="monospace")
    return fig


def get_dnn_acc(data):
    train_acc = pd.DataFrame(data["accuracy"], columns=["accuracy"])
    val_acc = pd.DataFrame(data["val_accuracy"], columns=["accuracy"])
    train_acc["type"] = "train"
    val_acc["type"] = "validation"
    acc_df = pd.concat([train_acc, val_acc])
    fig = px.line(
        acc_df, y="accuracy", color="type", title="DNN Learning Curve Accuracy"
    )
    fig.update_layout(font_family="monospace")
    fig.update_xaxes(title="epoch")
    return fig


def get_dnn_loss(data):
    train_loss = pd.DataFrame(data["loss"], columns=["loss"])
    val_loss = pd.DataFrame(data["val_loss"], columns=["loss"])
    train_loss["type"] = "train"
    val_loss["type"] = "validation"
    acc_df = pd.concat([train_loss, val_loss])
    fig = px.line(acc_df, y="loss", color="type", title="DNN Learning Curve Loss")
    fig.update_layout(font_family="monospace")
    fig.update_xaxes(title="epoch")
    return fig


def get_temporal_data(df):
    dates = pd.to_datetime(df.created_at, infer_datetime_format=True)
    sents = df.sentiment.replace(["negative", "neutral", "positive"], [-1, 0, 1])
    date_df = pd.DataFrame(zip(dates, sents), columns=["date", "sentiment"])
    group_df = date_df.groupby(pd.Grouper(key="date", freq="H")).mean()
    group_df.dropna(inplace=True)
    fig = px.area(
        group_df,
        x=group_df.index,
        y="sentiment",
        width=1600,
        title="Hourly Mean Sentiment (Positive=1, Neutral=0, Negative=-1)",
    )
    fig.update_yaxes(title="mean sentiment")
    fig.update_layout(font_family="monospace")
    return fig


def get_temporal_sum_data(df):
    dates = pd.to_datetime(df.created_at, infer_datetime_format=True).to_numpy()
    date_df = pd.DataFrame(dates, columns=["date"])
    group_df = date_df.groupby(pd.Grouper(key="date", freq="H"))
    group_df = group_df.size().reset_index(name="count")
    group_df.dropna(inplace=True)
    fig = px.histogram(
        group_df,
        x=group_df.date,
        y="count",
        width=1600,
        title="Hourly Tweet Count",
        nbins=int(group_df.size),
    )
    fig.update_yaxes(title="count")
    fig.update_layout(font_family="monospace")
    return fig


def get_bar_plot(data, range):
    _, y_pred = data
    sent_counts = (
        pd.DataFrame(y_pred, columns=["sentiment"]).sentiment.value_counts(
            normalize=True
        )
        * 100
    )
    data = pd.DataFrame(
        zip(sent_counts.keys(), sent_counts.values), columns=["sentiment", "percentage"]
    )
    fig = px.bar(
        data,
        x="sentiment",
        y="percentage",
        color="sentiment",
        color_discrete_map={
            "negative": "orangered",
            "positive": "limegreen",
            "neutral": "blue",
        },
        title="SVM Predicted Sentiment Breakdown",
    )
    fig.update_yaxes(range=range)
    fig.update_layout(font_family="monospace")
    return fig


def get_bar_plot_dnn(data, range):
    _, y_pred = data
    sent_counts = y_pred.value_counts(normalize=True) * 100
    data = pd.DataFrame(
        zip(sent_counts.keys(), sent_counts.values), columns=["sentiment", "percentage"]
    )
    fig = px.bar(
        data,
        x="sentiment",
        y="percentage",
        color="sentiment",
        color_discrete_map={
            "negative": "orangered",
            "positive": "limegreen",
            "neutral": "blue",
        },
        title="DNN Predicted Sentiment Breakdown",
    )
    fig.update_yaxes(range=range)
    fig.update_layout(font_family="monospace")
    return fig


app.layout = html.Div(
    children=[
        html.H1(children="Twitter data dashboard"),
        html.Div(
            children="""
        Data Visualisation (DV) Common Module Assessment (CMA)
    """
        ),
        html.Div(
            children="""
    """
        ),
        html.Br(),
        html.H3(children="Input Data"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="pie-graph",
                                    figure=get_pie_chart(df.sentiment),
                                ),
                            ]
                        )
                    ),
                    width=5,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="wordCloud",
                                    figure=get_word_cloud(
                                        " ".join(df.text.str.split(expand=True).stack())
                                    ),
                                )
                            ]
                        )
                    ),
                    width=5,
                ),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="geoPlot",
                                    figure=get_geo_plot(df),
                                ),
                            ]
                        )
                    ),
                    width=8,
                )
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="temporal",
                                    figure=get_temporal_data(df),
                                ),
                            ]
                        )
                    ),
                    width=10,
                )
            ],
            justify="center",
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="likes",
                                    figure=get_temporal_sum_data(df),
                                ),
                            ]
                        )
                    ),
                    width=10,
                )
            ],
            justify="center",
        ),
        html.Br(),
        html.H3(children="SVM Performance"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="svm-lc",
                                    figure=get_svm_lc(svm_lc),
                                ),
                            ]
                        )
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="svm_cm",
                                    figure=get_confusion_matrix(
                                        svm, "SVM Confusion Matrix"
                                    ),
                                ),
                            ]
                        )
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="svm_bar",
                                    figure=get_bar_plot(svm, [25, 36]),
                                ),
                            ]
                        )
                    ),
                    width=4,
                ),
            ],
            justify="center",
        ),
        html.Br(),
        html.H3(children="DNN Performance"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="dnn-acc",
                                    figure=get_dnn_acc(dnn_history),
                                ),
                            ]
                        )
                    ),
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="dnn-loss",
                                    figure=get_dnn_loss(dnn_history),
                                ),
                            ]
                        )
                    ),
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="dnn_cm",
                                    figure=get_confusion_matrix(
                                        dnn, "DNN Confusion Matrix"
                                    ),
                                ),
                            ]
                        )
                    ),
                ),
            ],
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id="svm_bar_2",
                                    figure=get_bar_plot_dnn(dnn, [25, 40]),
                                ),
                            ]
                        )
                    ),
                    width=4,
                ),
            ],
            justify="center",
        ),
    ]
)


@app.callback(
    Output("wordCloud", "figure"),
    [Input("pie-graph", "clickData"), Input("wordCloud", "clickData")],
    prevent_initial_call=True,
)
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


if __name__ == "__main__":
    app.run_server(debug=True)
