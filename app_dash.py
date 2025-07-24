
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import base64
import io
import json

# ==============================================================================
# 1. App Initialization
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# ==============================================================================
# 2. App Layout
# ==============================================================================
app.layout = dbc.Container([
    # Store data in the browser
    dcc.Store(id='store-batch-data', data={}),  # {'batch_name': df.to_json(), ...}
    dcc.Store(id='store-quality-data', data=None), # df.to_json()
    dcc.Store(id='store-pv-name', data='Value'),
    dcc.Store(id='store-time-offsets', data={}), # {'batch_name': offset_value, ...}
    
    # App Title
    html.H1("배치 데이터 분석 플랫폼 (Dash 기반)", className="my-4"),

    # Main Layout
    dbc.Row([
        # --- Left Column: Controls ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Step 1: 배치 데이터 입력"),
                dbc.CardBody([
                    dbc.Label("시간 간격 선택:"),
                    dcc.Dropdown(
                        id='dropdown-time-unit',
                        options=[
                            {'label': "1 minute", 'value': 1},
                            {'label': "5 minutes", 'value': 5},
                            {'label': "10 minutes", 'value': 10},
                            {'label': "30 minutes", 'value': 30},
                            {'label': "1 hour", 'value': 60},
                        ],
                        value=5,
                        clearable=False
                    ),
                    dbc.Label("배치명:", className="mt-3"),
                    dbc.Input(id='input-batch-name', placeholder="예: IBC24001", type="text"),
                    
                    dbc.Label("프로세스 변수(PV) 이름:", className="mt-3"),
                    dbc.Input(id='input-pv-name', placeholder="예: Temperature", type="text"),

                    dbc.Label("데이터 붙여넣기:", className="mt-3"),
                    dbc.Textarea(id='textarea-data', placeholder="엑셀의 한 열을 복사하여 여기에 붙여넣으세요...", style={'height': 200}),
                    
                    dbc.Button("배치 추가", id='button-add-batch', color="primary", className="mt-3 w-100"),
                    html.Div(id='output-batch-add-status', className="mt-2")
                ])
            ]),
            dbc.Card(id='card-batch-list', className="mt-4", children=[
                dbc.CardHeader("추가된 배치 목록"),
                dbc.CardBody(id='div-batch-list', children=[
                    # Batch list will be populated by a callback
                ]),
                dbc.CardFooter([
                    dbc.Button("모든 배치 초기화", id='button-clear-all-batches', color="danger", outline=True, size="sm", className="w-100")
                ])
            ]),
            html.Hr(),
            dbc.Card(id='card-quality-data', className="mt-4", children=[
                dbc.CardHeader("Step 2: 품질 데이터 업로드"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-quality-data',
                        children=html.Div(['파일을 드래그하거나 ', html.A('여기를 클릭해서 선택하세요')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                        },
                        multiple=False
                    ),
                    html.Div([
                        dash_table.DataTable(
                            id='datatable-quality',
                            editable=True,
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                        ),
                        dbc.Row([
                            dbc.Col(dbc.Button("변경사항 저장", id="button-save-quality-changes", color="primary", size="sm", className="mt-2 w-100"), width=6),
                            dbc.Col(dbc.Button("품질 데이터 초기화", id="button-reset-quality", color="secondary", size="sm", className="mt-2 w-100"), width=6)
                        ], id="div-quality-buttons", style={'display': 'none'})
                    ])
                ])
            ])
        ], md=4),
        
        # --- Right Column: Outputs ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("배치 프로파일"),
                dbc.CardBody([
                    dcc.Graph(id='graph-batch-profile', style={'height': '500px'})
                ])
            ]),
            dbc.Card(className="mt-4", children=[
                dbc.CardHeader("그래프 조정"),
                dbc.CardBody([
                    dbc.Accordion(id='accordion-graph-controls', start_collapsed=True, children=[
                        dbc.AccordionItem(
                            title="개별 배치 시간 이동",
                            children=html.Div(id='div-time-shift-sliders')
                        )
                    ])
                ])
            ])
        ], md=8),
    ])
], fluid=True)

# ==============================================================================
# 3. Callbacks
# ==============================================================================

# --- Callback to add new batch data ---
@app.callback(
    [Output('store-batch-data', 'data'),
     Output('store-pv-name', 'data'),
     Output('output-batch-add-status', 'children'),
     Output('input-batch-name', 'value'),
     Output('textarea-data', 'value')],
    [Input('button-add-batch', 'n_clicks')],
    [State('dropdown-time-unit', 'value'),
     State('input-batch-name', 'value'),
     State('input-pv-name', 'value'),
     State('textarea-data', 'value'),
     State('store-batch-data', 'data'),
     State('store-pv-name', 'data')]
)
def add_batch_data(n_clicks, interval, batch_name, pv_name, data_paste_area, existing_batches, current_pv_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update

    if not batch_name or not data_paste_area:
        alert = dbc.Alert("배치명과 데이터를 모두 입력해주세요.", color="warning", dismissable=True)
        return dash.no_update, dash.no_update, alert, dash.no_update, dash.no_update

    try:
        lines = data_paste_area.strip().split('\n')
        values = [float(line.strip()) for line in lines]
        time_index = [i * interval for i in range(len(values))]
        df = pd.DataFrame({'Time': time_index, 'Value': values})
        
        new_batches = existing_batches.copy()
        new_batches[batch_name] = df.to_json(date_format='iso', orient='split')
        
        # Update PV name only if a new one is provided
        new_pv_name = pv_name if pv_name else current_pv_name

        alert = dbc.Alert(f"배치 '{batch_name}'가 성공적으로 추가되었습니다.", color="success", dismissable=True)
        # Clear input fields after successful submission
        return new_batches, new_pv_name, alert, '', ''

    except ValueError:
        alert = dbc.Alert("데이터에 숫자가 아닌 값이 포함되어 있습니다. 숫자 데이터만 입력해주세요.", color="danger", dismissable=True)
        return dash.no_update, dash.no_update, alert, dash.no_update, dash.no_update
    except Exception as e:
        alert = dbc.Alert(f"오류가 발생했습니다: {e}", color="danger", dismissable=True)
        return dash.no_update, dash.no_update, alert, dash.no_update, dash.no_update

# --- Callback to update batch list display ---
@app.callback(
    Output('div-batch-list', 'children'),
    Input('store-batch-data', 'data')
)
def update_batch_list(batch_data):
    if not batch_data:
        return dbc.Alert("추가된 배치가 없습니다.", color="secondary")
    
    batch_list = []
    for batch_name in reversed(list(batch_data.keys())):
        try:
            df = pd.read_json(batch_data[batch_name], orient='split')
            list_item = dbc.Row([
                dbc.Col(f"**{batch_name}** ({len(df)} points)", width='auto'),
                dbc.Col(dbc.Button("삭제", 
                                   id={'type': 'delete-batch-button', 'index': batch_name}, 
                                   color="danger", 
                                   outline=True, 
                                   size="sm"), width='auto')
            ], justify="between", align="center", className="mb-2")
            batch_list.append(list_item)
        except Exception:
            # Handle cases where data might be malformed, though unlikely
            continue
            
    return batch_list

# --- Callback to clear all batches ---
@app.callback(
    Output('store-batch-data', 'data', allow_duplicate=True),
    Input('button-clear-all-batches', 'n_clicks'),
    prevent_initial_call=True
)
def clear_all_batches(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    return {}

# --- Callback for individual batch deletion ---
@app.callback(
    Output('store-batch-data', 'data', allow_duplicate=True),
    Input({'type': 'delete-batch-button', 'index': dash.ALL}, 'n_clicks'),
    State('store-batch-data', 'data'),
    prevent_initial_call=True
)
def delete_batch(n_clicks, batch_data):
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered or all(c is None for c in n_clicks):
        return dash.no_update
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    batch_to_delete = json.loads(button_id)['index']
    
    if batch_to_delete in batch_data:
        new_batch_data = batch_data.copy()
        del new_batch_data[batch_to_delete]
        return new_batch_data
        
    return dash.no_update

# --- Callback to update graph and time shift sliders ---
@app.callback(
    [Output('graph-batch-profile', 'figure'),
     Output('div-time-shift-sliders', 'children')],
    [Input('store-batch-data', 'data'),
     Input('store-time-offsets', 'data'),
     Input('store-pv-name', 'data')]
)
def update_graph_and_sliders(batch_data, time_offsets, pv_name):
    if not batch_data:
        fig = go.Figure()
        fig.update_layout(
            title_text='좌측 메뉴에서 배치 데이터를 추가하면 여기에 그래프가 표시됩니다.',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
        return fig, html.Div("표시할 배치가 없습니다.")

    fig = FigureResampler(go.Figure())
    max_time_list = []

    for batch_name, json_df in batch_data.items():
        df = pd.read_json(json_df, orient='split')
        offset = time_offsets.get(batch_name, 0.0)
        
        x_data = df['Time'] + offset
        y_data = df['Value']
        
        fig.add_trace(go.Scattergl(
            name=batch_name,
            mode='lines', # Use 'lines' for performance on large datasets
        ), hf_x=x_data, hf_y=y_data)

        if not df.empty:
            max_time_list.append(df['Time'].max())

    max_time = max(max_time_list) if max_time_list else 100

    fig.update_layout(
        title_text='그래프 하단의 바를 드래그하여 범위를 조절하세요',
        xaxis_title='Time (Shifted)',
        yaxis_title=pv_name,
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    sliders = []
    for batch_name in batch_data.keys():
        slider = html.Div([
            html.Label(f"'{batch_name}' 이동"),
            dcc.Slider(
                id={'type': 'time-shift-slider', 'index': batch_name},
                min=-max_time,
                max=max_time,
                value=time_offsets.get(batch_name, 0.0),
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], className="mb-3")
        sliders.append(slider)
        
    return fig, sliders

# --- Callback to store time shift slider values ---
@app.callback(
    Output('store-time-offsets', 'data'),
    Input({'type': 'time-shift-slider', 'index': dash.ALL}, 'value'),
    State('store-time-offsets', 'data'),
)
def store_slider_values(values, offsets):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    triggered_id = json.loads(triggered_id_str)
    batch_name = triggered_id['index']
    slider_value = ctx.triggered[0]['value']
    
    new_offsets = offsets.copy()
    new_offsets[batch_name] = slider_value
    
    return new_offsets

# --- Callback to upload and parse quality data ---
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

@app.callback(
    Output('store-quality-data', 'data', allow_duplicate=True),
    Input('upload-quality-data', 'contents'),
    State('upload-quality-data', 'filename'),
    prevent_initial_call=True
)
def update_quality_data_store(list_of_contents, list_of_names):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names)
        if df is not None:
            return df.to_json(date_format='iso', orient='split')
    return dash.no_update

# --- Callback to display quality data table from store ---
@app.callback(
    [Output('datatable-quality', 'data'),
     Output('datatable-quality', 'columns'),
     Output('div-quality-buttons', 'style')],
    Input('store-quality-data', 'data')
)
def update_table_from_store(json_data):
    if json_data is None:
        return [], [], {'display': 'none'}
    
    df = pd.read_json(json_data, orient='split')
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')
    return data, columns, {'display': 'flex', 'margin-top': '10px'}

# --- Callback to save table changes to store ---
@app.callback(
    Output('store-quality-data', 'data', allow_duplicate=True),
    Input('button-save-quality-changes', 'n_clicks'),
    State('datatable-quality', 'data'),
    prevent_initial_call=True
)
def save_quality_changes_to_store(n_clicks, table_rows):
    if n_clicks is None or not table_rows:
        return dash.no_update
    
    df = pd.DataFrame(table_rows)
    return df.to_json(date_format='iso', orient='split')

# --- Callback to reset quality data ---
@app.callback(
    Output('store-quality-data', 'data', allow_duplicate=True),
    Input('button-reset-quality', 'n_clicks'),
    prevent_initial_call=True
)
def reset_quality_data(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    return None


if __name__ == '__main__':
    app.run(debug=True) 