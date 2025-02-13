import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_excel("Card_list_20240709.xlsx")
df['Date Bought'] = pd.to_datetime(df['Date Bought']).dt.date
df['Profit'] = df['AVG'] - df['Price Bought']
owner_name = "Luke"

# Colors for styling
colors = {
    'background': '#1a1a1a',  # Slightly lighter for better contrast
    'text': '#DDDDDD',  # Softer white text
    'highlight_yellow': '#FFD700',
    'highlight_green': '#00FF7F',  # Brighter green for visibility
    'highlight_light_blue': '#ADD8E6',
    'table_header': '#222222',  # Darker header background
    'row_odd': '#1e1e1e',  # Alternating row color
    'row_selected': '#444444'  # Highlight selected row
}


# Initialize app
app = dash.Dash()

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'margin': '10px', 'padding': '10px'}, children=[
    html.H1(
    f"{owner_name}'s Card Collection",
    style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '50px', 'fontWeight': 'bold'}),
    html.Br(),

    # Date Picker
    html.Div(children=[
        html.H3("Filter by Date Bought:", style={'color': colors['text'], 'fontSize': '24px', 'fontWeight': 'bold'}),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=df['Date Bought'].min(),
            end_date=df['Date Bought'].max(),
            display_format='DD-MM-YYYY',
            style={
                'backgroundColor': '#222222',  # Dark background
                'color': 'white',  # White text
                'border': '1px solid #666666',  # Soft gray border (less contrast)
                'borderRadius': '6px',  # Rounded edges
                'padding': '6px',
                'fontSize': '18px',
                'boxShadow': '0px 2px 5px rgba(255, 255, 255, 0.2)'  # Soft glow effect
            }
        )
    ]),
    html.Br(),

    # Summary Text
    html.Div(style={'textAlign': 'center', 'color': colors['text']}, children=[
        html.Span([
            f"{owner_name} currently has ",
            html.Span(id='card-count', style={'color': colors['highlight_light_blue']}),
            " cards in their collection worth a total of ",
            html.Span(id='collection-value', style={'color': colors['highlight_green']}),
            "€"
        ], style={'fontSize': '20px'}),
        html.Br(),
        html.Span([
            "The most valuable card is ",
            html.Span(id='most-valuable-card', style={'color': colors['highlight_yellow']}),
            " worth approximately ",
            html.Span(id='most-valuable-price', style={'color': colors['highlight_green']}),
            "€"
        ], style={'fontSize': '20px'})
    ]),
    html.Br(),

    # Dropdown Filters
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
        html.Div(children=[
            html.H3("Filter by Set Name:", style={'color': colors['text'], 'fontSize': '24px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='set-filter',
                options=[{'label': s, 'value': s} for s in sorted(df['Set Name'].unique())],
                multi=True,
                searchable=True,
                placeholder="Select Set Name(s)",
                style={
                    'backgroundColor': '#222222',  # Dark background
                    'color': 'white',  # White text inside dropdown
                    'border': '0.5px solid #FFD966',  # Lighter gold border, softer thickness
                    'padding': '8px',
                    'borderRadius': '6px',  # Slightly less rounded for a modern feel
                    'fontSize': '18px',
                    'width': '300px',
                    'boxShadow': '0px 2px 5px rgba(255, 217, 102, 0.3)'  # Soft gold shadow
                }
            )
        ]),

        html.Div(children=[
            html.H3("Filter by Language:", style={'color': colors['text'], 'fontSize': '24px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='language-filter',
                options=[{'label': l, 'value': l} for l in sorted(df['Language'].unique())],
                multi=True,
                searchable=True,
                placeholder="Select Language(s)",
                style={
                    'backgroundColor': '#222222',  # Dark background
                    'color': 'white',  # White text inside dropdown
                    'border': '0.5px solid #66FF99',  # Softer green border
                    'padding': '8px',
                    'borderRadius': '6px',  # Slightly less rounded for a modern feel
                    'fontSize': '18px',
                    'width': '300px',
                    'boxShadow': '0px 2px 5px rgba(102, 255, 153, 0.3)'  # Soft gold shadow
                }
            )
        ])
    ]),

    html.Br(),

    # Bar Charts
    html.Div(
        style={
            'display': 'flex',
            'justifyContent': 'space-around',  # Ensures spacing between charts
            'alignItems': 'flex-start',  # Aligns them properly
            'flexWrap': 'wrap',  # Prevents overflow
            'width': '100%'
        },
        children=[
            html.Div(
                style={
                    'width': '70%',  
                    'minWidth': '350px',  # Ensures responsiveness
                    'paddingRight': '8px',  # Adds spacing between the two charts
                    'boxSizing': 'border-box',  # Prevents border from affecting width
                },
                children=[
                    html.H2("Top 10 Sets by Volume", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='set-bar-chart',
                        style={
                            'width': '100%',
                            'borderRadius': '6px',
                            'border': '1px solid #666666',
                            'boxShadow': '0px 2px 5px rgba(255, 255, 255, 0.2)',
                            'padding': '10px'
                        }
                    )
                ]
            ),
          
            html.Div(
                style={
                    'width': '25%',  # Reduce width to prevent overlap
                    'minWidth': '350px',
                    'paddingLeft': '8px',
                    'paddingRight': '10px',  # Adds spacing between the two charts
                    'boxSizing': 'border-box',
                },
                children=[
                    html.H2("Cards per Language", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='language-bar-chart',
                        style={
                            'width': '100%',
                            'borderRadius': '6px',
                            'border': '1px solid #666666',
                            'boxShadow': '0px 2px 5px rgba(255, 255, 255, 0.2)',
                            'padding': '10px'
                        }
                    )
                ]
            )
        ]
    ),
    html.Br(),
    html.Br(),
    html.Br(),

    # Table and Card Preview
    html.Div(style={'display': 'flex'}, children=[
        html.Div(style={'width': '70%'}, children=[
            html.H2("Full Collection", style={'textAlign': 'center', 'color': colors['text']}),

            dash_table.DataTable(
                id='card-table',
                columns=[
                    {'name': 'Card Name', 'id': 'Card Name'},
                    {'name': 'Set Name', 'id': 'Set Name'},
                    {'name': 'Language', 'id': 'Language'},
                    {'name': 'Date Bought', 'id': 'Date Bought'},
                    {'name': 'AVG', 'id': 'AVG', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Profit', 'id': 'Profit', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                filter_action="native",
                sort_action="native",
                page_size=20,
                row_selectable='single',

                # 🔻 Fix filter box background & text visibility
                style_filter={
                    'backgroundColor': '#222222',  # Dark background for filter boxes
                    'color': 'white',  # White text so it's visible
                    'border': '1px solid #444444',  # Dark gray border (no yellow)
                    'padding': '6px',
                    'borderRadius': '6px',
                },

                style_table={
                    'backgroundColor': colors['background'],
                    'color': colors['text'],
                    'borderRadius': '6px',
                    'border': '1px solid #666666',
                    'boxShadow': '0px 2px 5px rgba(255, 255, 255, 0.2)',
                    'padding': '10px'
                },
                style_header={'backgroundColor': colors['table_header'], 'color': colors['text'], 'fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': colors['background'],
                    'color': colors['text'],
                    'padding': '10px',
                    'border': '1px solid #222222'  # Subtle cell borders
                },
            )
        ]),

        # Card Preview Section
        html.Div(style={'width': '30%', 'textAlign': 'center'}, children=[
            html.H2("Card Preview", style={'color': colors['text']}),
            html.Img(id='card-preview', style={'width': '75%', 'height': 'auto'})
        ])
    ]),  # <-- Closing the table and card preview section properly

    html.Br(),
    html.Br(),

    # Chat with Collection (Placed Below the Table and Card Preview)
    html.Div(style={'marginTop': '30px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}, children=[
        html.H2("Talk to the Collection", style={'color': colors['text'], 'fontSize': '28px', 'fontWeight': 'bold'}),

        # User Input for ChatGPT
        html.Div(style={'width': '70%', 'display': 'flex', 'justifyContent': 'center'}, children=[
            dcc.Input(
                id='user-input',
                type='text',
                placeholder="Ask something about your collection...",
                debounce=True,  # Ensures input is processed only after the user stops typing
                style={
                    'width': '70%',
                    'padding': '10px',
                    'fontSize': '18px',
                    'borderRadius': '6px',
                    'backgroundColor': '#222222',
                    'color': 'white',
                    'border': '1px solid #666666'
                }
            ),
            html.Button(
                "Ask", id="ask-button", n_clicks=0,
                style={
                    'marginLeft': '10px',
                    'padding': '10px 20px',
                    'fontSize': '18px',
                    'borderRadius': '6px',
                    'backgroundColor': '#FFD700',
                    'color': '#111111',
                    'border': 'none',
                    'cursor': 'pointer'
                }
            )
        ]),

        html.Br(),

        # ChatGPT Response & Generated Chart Side by Side
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'width': '80%'}, children=[
            # ChatGPT Response
            html.Div(style={'width': '48%', 'paddingRight': '10px'}, children=[
                dcc.Markdown(id="chat-response", style={'color': colors['text'], 'fontSize': '18px', 'textAlign': 'left', 'whiteSpace': 'pre-wrap'})
            ]),

            # Generated Chart
            html.Div(style={'width': '48%', 'paddingLeft': '10px'}, children=[
                dcc.Graph(id="generated-chart")  # Dynamic Chart Placeholder
            ])
        ])
    ])
    
])



# Callback to filter table
@app.callback(
    Output('card-table', 'data'),
    [Input('set-filter', 'value'), Input('language-filter', 'value'), Input('date-picker', 'start_date'), Input('date-picker', 'end_date')]
)
def update_filtered_table(selected_sets, selected_languages, start_date, end_date):
    filtered_df = df.copy()
    if selected_sets:
        filtered_df = filtered_df[filtered_df['Set Name'].isin(selected_sets)]
    if selected_languages:
        filtered_df = filtered_df[filtered_df['Language'].isin(selected_languages)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date Bought'] >= pd.to_datetime(start_date).date()) &
                                  (filtered_df['Date Bought'] <= pd.to_datetime(end_date).date())]
    return filtered_df.to_dict('records')

# Callback to update bar charts
@app.callback(
    [Output('set-bar-chart', 'figure'), Output('language-bar-chart', 'figure')],
    [Input('card-table', 'data')]
)
def update_bar_charts(table_data):
    df_filtered = pd.DataFrame(table_data)
    
    # Generate the Set Bar Chart
    set_chart = px.bar(
        df_filtered.groupby('Set Name').size().reset_index(name='Count'),
        x='Set Name', y='Count',
        title="Top 10 Sets by Volume",
        color_discrete_sequence=["#FFD700"]  # Gold bars for better contrast
    )

    # Generate the Language Bar Chart
    language_chart = px.bar(
        df_filtered.groupby('Language').size().reset_index(name='Count'),
        x='Language', y='Count',
        title="Cards per Language",
        color_discrete_sequence=["#00FF7F"]  # Spring green for clear differentiation
    )

    # Style Improvements
    for chart in [set_chart, language_chart]:
        chart.update_layout(
            plot_bgcolor=colors['background'],  # Match background to dark theme
            paper_bgcolor=colors['background'],  # Consistent background
            font_color=colors['text'],  # Make text visible
            xaxis=dict(
                showgrid=False,  # Remove distracting grid lines
                tickangle=-45,  # Rotate x-axis labels for readability
                title_font_size=18,  # Bigger axis title
                tickfont_size=14,  # Readable tick labels
            ),
            yaxis=dict(
                showgrid=False,  # Cleaner chart
                title_font_size=18
            ),
            title_font_size=22,  # Larger title
            margin=dict(l=20, r=20, t=50, b=50)  # Improve spacing
        )
        
        # Make bars more visually appealing
        chart.update_traces(
            marker=dict(
                line=dict(width=1.5, color='white'),  # Add white outline for better visibility
                opacity=0.9  # Make bars slightly transparent for a modern look
            )
        )

    return set_chart, language_chart


# Callback to update card preview
@app.callback(
    Output('card-preview', 'src'),
    Input('card-table', 'selected_rows'),
    State('card-table', 'data')
)
def update_card_preview(selected_rows, table_data):
    if selected_rows and table_data:
        row = table_data[selected_rows[0]]
        return f"https://assets.pokemon.com/static-assets/content-assets/cms2/img/cards/web/{row['Set Code']}/{row['Set Code']}_EN_{row['Card Number']}.png"
    return ""

@app.callback(
    [Output('card-count', 'children'), Output('collection-value', 'children'), Output('most-valuable-card', 'children'), Output('most-valuable-price', 'children')],
    [Input('card-table', 'data')]
)
def update_summary_text(table_data):
    df_filtered = pd.DataFrame(table_data)
    if df_filtered.empty:
        return 0, 0, "None", 0
    return len(df_filtered), f"{df_filtered['AVG'].sum():.2f}", df_filtered.loc[df_filtered['AVG'].idxmax(), 'Card Name'], f"{df_filtered['AVG'].max():.2f}"

import openai
import plotly.express as px

# Initialize OpenAI client
client = openai.OpenAI(api_key="sk-proj-wuHCnmIjfT2PqcXbTHnsT3BlbkFJv8i91ZabcLFBJxV0HdbK")  # ✅ Updated for OpenAI v1.0+

@app.callback(
    [Output("chat-response", "children"), Output("generated-chart", "figure")],
    Input("ask-button", "n_clicks"),
    State("user-input", "value"),
    State("card-table", "data"),  # Get the filtered dataset
    prevent_initial_call=True
)
def ask_chatgpt(n_clicks, user_question, filtered_data):
    if not user_question:
        return "Please enter a question.", px.scatter(title="No Data")

    # Convert the filtered data from the table into a DataFrame
    filtered_df = pd.DataFrame(filtered_data)

    if filtered_df.empty:
        return "There are no cards matching your current filters.", px.scatter(title="No Data")

    # Summarize the number of cards per language
    language_counts = filtered_df["Language"].value_counts().to_dict()

    # Calculate earnings per set (sum of Profit column)
    earnings_per_set = filtered_df.groupby("Set Name")["Profit"].sum().reset_index()
    earnings_per_set = earnings_per_set.sort_values(by="Profit", ascending=False)

    # Create a summary of the dataset for ChatGPT
    dataset_summary = f"""
    You are assisting with questions about a Pokémon card collection.
    This dataset contains **all** available data, and all user queries refer to this dataset only.

    ### Language Breakdown:
    """
    for lang, count in language_counts.items():
        dataset_summary += f"- {count} cards in {lang}\n"

    dataset_summary += "\n### Earnings Per Set:\n"

    # Include earnings for ALL sets (not limited to top 10)
    for _, row in earnings_per_set.iterrows():
        dataset_summary += f"- {row['Set Name']} earned a total profit of {row['Profit']:.2f}€.\n"

    dataset_summary += "\n### Dataset Columns Available:\n"
    dataset_summary += "- Card Name, Set Name, Language, Date Bought, Price Bought, LOW, TREND, AVG, Profit (AVG - Price Bought).\n"
    dataset_summary += "\n**Use this dataset to answer all user questions.**"
    dataset_summary += f"\n\nThe user is asking: {user_question}"



    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                You are an expert in Pokémon card collection analysis.
                Assume the dataset provided contains **all** available data.
                Do not ask the user for more data. **All questions refer only to this dataset.**
                If the user asks for a ranking (e.g., "top 10"), only then should you limit the response.
                Otherwise, analyze all available data.
                """},
                {"role": "user", "content": dataset_summary}
            ]
        )



        answer = response.choices[0].message.content


        # Generate a relevant chart based on the query
        chart_figure = generate_chart_from_query(user_question, filtered_df)

        return f"**ChatGPT:**\n\n{answer}", chart_figure

    except Exception as e:
        return f"❌ Error: {str(e)}", px.scatter(title="No Data")


def generate_chart_from_query(user_question, df):
    """ Generates a chart based on the user's query and filtered dataset. """
    
    # Convert user input to lowercase for easier matching
    question_lower = user_question.lower()

    # 🔹 If the user asks for sum of profits per set
    if "sum of profits" in question_lower or "total profit" in question_lower:
        # Compute total profit per set (do not limit to top 10 unless specified)
        earnings_per_set = df.groupby("Set Name")["Profit"].sum().reset_index()
        earnings_per_set = earnings_per_set.sort_values(by="Profit", ascending=False)

        # Check if the user specifically asks for "top 10"
        if "top 10" in question_lower or "max 10" in question_lower:
            earnings_per_set = earnings_per_set.head(10)

        fig = px.bar(
            earnings_per_set,
            x="Set Name",
            y="Profit",
            title="Total Profit per Set (€)",
            labels={"Profit": "Total Profit (€)", "Set Name": "Set Name"}
        )

    elif "profit" in question_lower:
        # Show profit distribution
        fig = px.histogram(df, x="Profit", title="Profit Distribution of Cards", labels={"Profit": "Profit (€)"})
    
    elif "language" in question_lower:
        # Show the number of cards per language
        fig = px.pie(df, names="Language", title="Card Count per Language")
    
    elif "date" in question_lower:
        # Show number of cards bought over time
        fig = px.line(df.groupby("Date Bought").size().reset_index(name="Count"),
                      x="Date Bought", y="Count", title="Cards Bought Over Time",
                      labels={"Count": "Number of Cards", "Date Bought": "Date"})
    
    else:
        # Default chart if no match is found
        fig = px.scatter(title="No specific chart found for your question. Try asking about price, profit, or languages.")

    # Apply consistent styling
    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig


server = app.server  # Required for Render deployment

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)

#if __name__ == '__main__':
#    app.run_server(debug=True)