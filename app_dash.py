
import dash
from dash import dcc, html, Input, Output, State, dash_table, MATCH, ALL, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import base64
import io
import json
import pickle
import datetime

from sklearn.cross_decomposition import PLSRegression
# import joblib # Not used yet

# ==============================================================================
# 1. App Initialization
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# ==============================================================================
# 2. App Layout
# ==============================================================================
app.layout = dbc.Container([
    # --- Data Stores ---
    # Raw Data
    dcc.Store(id='store-batch-data', data={}),  # {'batch_name': df.to_json(), ...}
    dcc.Store(id='store-quality-data', data=None), # df.to_json()
    dcc.Store(id='store-pv-name', data='Value'),
    dcc.Store(id='store-time-offsets', data={}), # {'batch_name': offset_value, ...}

    # Analysis Settings & Results
    dcc.Store(id='store-analysis-range', data={'start': None, 'end': None}),
    dcc.Store(id='store-golden-batch-settings', data={'target_col': None, 'direction': '최대화', 'selected_batches': [], 'n_std': 2.0}),
    dcc.Store(id='store-aligned-data', data=None), # List of dicts with aligned data
    dcc.Store(id='store-pls-fingerprint', data=None), # To check if PLS model is outdated
    dcc.Store(id='store-pls-model', data=None), # Serialized PLS model
    dcc.Store(id='store-optimized-profile', data=None), # Optimized profile data
    dcc.Store(id='store-golden-batch-analysis', data=None), # Store for golden batch analysis results
    dcc.Store(id='store-debug-log', data=[]), # New log store

    # --- App Title ---
    html.H1("배치 데이터 분석 플랫폼 (Dash 기반)", className="my-4"),

    # --- TABS ---
    dbc.Tabs(id="tabs-main", active_tab='tab-visualize', children=[
        # Tab 1: Already static, no changes needed
        dbc.Tab(label="📊 데이터 시각화 및 정렬", tab_id='tab-visualize', children=[
            dbc.Row([
                # LEFT COLUMN FOR CONTROLS
                dbc.Col(width=4, children=[
                    # Batch Data Input
                    dbc.Card([
                        dbc.CardHeader("Step 1: 배치 데이터 입력"),
                        dbc.CardBody([
                            dbc.Label("시간 간격 선택:"),
                            dcc.Dropdown(id='dropdown-time-unit',
                                         options=[{'label': f"{v} minute{'s' if v > 1 else ''}", 'value': v} for v in [1, 5, 10, 30, 60]],
                                         value=5, clearable=False),
                            dbc.Label("배치명:", className="mt-3"),
                            dbc.Input(id='input-batch-name', placeholder="예: IBC24001", type="text"),
                            dbc.Label("프로세스 변수(PV) 이름:", className="mt-3"),
                            dcc.Input(id='input-pv-name', placeholder="예: Temperature", type="text", className="form-control"),
                            dbc.Label("데이터 붙여넣기:", className="mt-3"),
                            dbc.Textarea(id='textarea-data', placeholder="엑셀의 한 열을 복사하여 여기에 붙여넣으세요...", style={'height': 150}),
                            dbc.Button("배치 추가", id='button-add-batch', color="primary", className="mt-3 w-100"),
                            html.Div(id='output-batch-add-status', className="mt-2")
                        ])
                    ]),
                    # Batch List
                    dbc.Card(className="mt-3", children=[
                        dbc.CardHeader("추가된 배치 목록"),
                        dbc.CardBody(id='div-batch-list-container', style={'maxHeight': '200px', 'overflowY': 'auto'}),
                        dbc.CardFooter(dbc.Button("모든 배치 초기화", id='button-clear-all-batches', color="danger", outline=True, size="sm", className="w-100"))
                    ]),
                    # Quality Data
                    dbc.Card(className="mt-3", children=[
                        dbc.CardHeader("Step 2: 품질 데이터 업로드"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-quality-data',
                                children=html.Div(['파일을 드래그하거나 ', html.A('클릭하여 선택')]),
                                style={
                                    'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 
                                    'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'
                                },
                                multiple=False
                            ),
                            html.Div(id='div-quality-data-table-container')
                        ]),
                        dbc.CardFooter(
                            dbc.Row([
                                dbc.Col(html.Pre(id='debug-save-action', style={'fontSize': 'small'})),
                                dbc.Col([
                                    dbc.Button("변경사항 저장", id="button-save-quality-changes", color="primary", size="sm", style={'display': 'none'}),
                                    dbc.Button("품질 데이터 초기화", id="button-reset-quality", color="secondary", size="sm", style={'display': 'none'}, className="ms-2")
                                ], width="auto")
                            ], justify="between")
                        )
                    ]),
                ]),
                # RIGHT COLUMN FOR GRAPH
                dbc.Col(width=8, children=[
                    html.Div(id='graph-adjustment-container'),
                    html.Div(id='analysis-range-container'),
                    dbc.Card(dbc.CardBody(dcc.Graph(id='main-graph-tab1')))
                ])
            ])
        ]),
        # Tab 2: Now contains its full, static layout, initially hidden
        dbc.Tab(label="🌟 골든 배치 분석", tab_id='tab-golden-batch', children=[
            html.Div(id='golden-batch-alert-wrapper'),
            html.Div(id='golden-batch-controls-wrapper', style={'display': 'none'}, children=[
                dbc.Row([
                    dbc.Col(width=4, children=[
                        dbc.Card([
                            dbc.CardHeader("골든 배치 분석 설정"),
                            dbc.CardBody([
                                dbc.Label("1. 품질 목표 선택"),
                                dcc.Dropdown(id='dropdown-quality-target', options=[]),
                                dbc.Label("2. 최적화 방향 선택", className="mt-3"),
                                dbc.RadioItems(id='radio-optimization-direction',
                                    options=[{'label': '최대화', 'value': 'max'}, {'label': '최소화', 'value': 'min'}],
                                    value='max', inline=True),
                                html.Hr(),
                                dbc.Label("3. 골든 배치 선택"),
                                # The Div is now just a container for the static Checklist
                                html.Div(id='div-golden-batch-selection', children=[
                                    dbc.Checklist(
                                        id='checklist-golden-batches',
                                        options=[],
                                        value=[],
                                        style={'maxHeight': '200px', 'overflowY': 'auto', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
                                    )
                                ]),
                                html.Hr(),
                                dbc.Button("골든 프로파일 생성", id='button-generate-golden-profile', color="primary", className="w-100"),
                            ])
                        ])
                    ]),
                    dbc.Col(width=8, children=[
                        dbc.Card([
                            dbc.CardHeader("분석 결과"),
                            dbc.CardBody([
                                dcc.Slider(id='slider-std-dev', min=0, max=5, step=0.1, value=3, marks={i: str(i) for i in range(6)}),
                                html.Div(id='golden-batch-graph-container')
                            ])
                        ])
                    ])
                ])
            ])
        ]),
        # Tab 3: Now contains its full, static layout, initially hidden
        dbc.Tab(label="📈 PLS 원인 분석", tab_id='tab-pls-analysis', children=[
            html.Div(id='pls-analysis-alert-wrapper'),
            html.Div(id='pls-analysis-controls-wrapper', style={'display': 'none'}, children=[
                dbc.Row([
                    dbc.Col(width=4, children=[
                        dbc.Card([
                            dbc.CardHeader("PLS 원인 분석 설정"),
                            dbc.CardBody([
                                dbc.Label("1. PLS 컴포넌트 수"),
                                dbc.Input(id='input-n-components', type='number', value=2, min=1, step=1),
                                html.Hr(),
                                dbc.Button("PLS 분석 실행", id='button-run-pls', color="primary", className="w-100"),
                                html.Hr(),
                                dbc.Button("이론적 최적 프로파일 제안", id='button-suggest-profile', color='success', className="w-100"),
                                dcc.Loading(id="loading-pls", children=html.Div(id='pls-status-output'))
                            ])
                        ])
                    ]),
                    dbc.Col(width=8, children=[
                         dbc.Card([
                            dbc.CardHeader("분석 결과: 회귀 계수(Regression Coefficient) 플롯"),
                            dbc.CardBody(id='pls-graph-container')
                        ])
                    ])
                ])
            ])
        ]),
    ]),
    html.Hr(),
    dbc.Card(dbc.CardBody([
        html.H4("디버그 출력", className="card-title"),
        html.Pre(id='debug-output')
    ]))
], fluid=True)

# ==============================================================================
# 3. Callbacks for Tab 1 (Data Visualization) - LEFT COLUMN
# ==============================================================================
# DELETE the now-redundant render_tab1_left_column callback
# It is replaced by the static layout definition.

@app.callback(
    [Output('store-batch-data', 'data'),
     Output('store-pv-name', 'data', allow_duplicate=True),
     Output('output-batch-add-status', 'children'),
     Output('input-batch-name', 'value'),
     Output('textarea-data', 'value')],
    Input('button-add-batch', 'n_clicks'),
    [State('dropdown-time-unit', 'value'),
     State('input-batch-name', 'value'),
     State('input-pv-name', 'value'),
     State('textarea-data', 'value'),
     State('store-batch-data', 'data'),
     State('store-pv-name', 'data')],
    prevent_initial_call=True
)
def add_batch_data(n_clicks, interval, batch_name, pv_name_input, data_paste, batches, current_pv):
    if not batch_name or not data_paste:
        return dash.no_update, dash.no_update, dbc.Alert("배치명과 데이터를 모두 입력해주세요.", color="warning", dismissable=True), dash.no_update, dash.no_update
    try:
        # Prevent adding duplicate batch names
        if batch_name in batches:
            return dash.no_update, dash.no_update, dbc.Alert(f"오류: 배치명 '{batch_name}'은 이미 존재합니다.", color="danger", dismissable=True), dash.no_update, dash.no_update

        lines = data_paste.strip().split('\n') # Changed from '\\n' to '\n'
        values = [float(line.strip().replace(',', '')) for line in lines if line.strip()]
        time_index = [i * interval for i in range(len(values))]
        df = pd.DataFrame({'Time': time_index, 'Value': values})
        
        new_batches = batches.copy()
        new_batches[batch_name] = df.to_json(orient='split')
        
        # Only update PV name if it's not already set, or if user provides a new one
        new_pv = pv_name_input if pv_name_input else (current_pv if current_pv else "Value")

        return new_batches, new_pv, dbc.Alert(f"배치 '{batch_name}' 추가 완료.", color="success", dismissable=True), "", ""
    except ValueError:
         return dash.no_update, dash.no_update, dbc.Alert("데이터에 숫자가 아닌 값이 포함되어 있습니다. 숫자 데이터만 입력해주세요.", color="danger", dismissable=True), batch_name, data_paste
    except Exception as e:
        return dash.no_update, dash.no_update, dbc.Alert(f"오류: {e}", color="danger", dismissable=True), dash.no_update, dash.no_update

@app.callback(
    Output('div-batch-list-container', 'children'),
    Input('store-batch-data', 'data')
)
def update_batch_list_ui(batches):
    if not batches:
        return "추가된 배치가 없습니다."
    
    return [
        dbc.Row([
            dbc.Col(f"{name} ({len(pd.read_json(df_json, orient='split'))} points)"),
            dbc.Col(dbc.Button("삭제", id={'type': 'delete-batch-btn', 'index': name}, color="danger", size="sm", outline=True), width="auto")
        ], className="mb-1 align-items-center") for name, df_json in batches.items()
    ]

@app.callback(
    Output('store-batch-data', 'data', allow_duplicate=True),
    Input({'type': 'delete-batch-btn', 'index': ALL}, 'n_clicks'),
    State('store-batch-data', 'data'),
    prevent_initial_call=True
)
def delete_batch(n_clicks, batches):
    # Filter out the None clicks which happen on initialization
    if not any(n_clicks):
        return dash.no_update

    triggered_id_str = dash.callback_context.triggered[0]['prop_id']
    if not triggered_id_str:
        return dash.no_update
    
    batch_to_delete = json.loads(triggered_id_str.split('.')[0])['index']
    
    new_batches = batches.copy()
    if batch_to_delete in new_batches:
        del new_batches[batch_to_delete]
    return new_batches

@app.callback(
    Output('store-batch-data', 'data', allow_duplicate=True),
    Input('button-clear-all-batches', 'n_clicks'),
    prevent_initial_call=True
)
def clear_all_batches(n_clicks):
    return {}

# --- DELETING ALL PROBLEMATIC COMPONENTS AND CALLBACKS ---
# Delete the 'store-global-time-range' from the app.layout
# Delete the 'update_global_time_range' callback
# Delete the 'update_right_column_controls' callback
# Delete the two interactive store update callbacks ('update_time_offsets_store', 'update_analysis_range_store')

# --- ADDING THE NEW, FINAL, AND STABLE CALLBACKS ---

# This callback creates/re-creates the control UIs ONLY when the BATCH LIST changes.
@app.callback(
    [Output('graph-adjustment-container', 'children'),
     Output('analysis-range-container', 'children')],
    [Input('store-batch-data', 'data')],
    [State('store-time-offsets', 'data'),
     State('store-analysis-range', 'data')]
)
def update_right_column_controls(batch_data, time_offsets, analysis_range):
    if not batch_data:
        return None, None

    batches = list(batch_data.keys())
    
    # Calculate a STABLE range for the controls based on RAW data, ignoring offsets.
    # This prevents re-rendering on drag.
    raw_times = []
    for name, df_json in batch_data.items():
        df = pd.read_json(df_json, orient='split')
        if not df.empty:
            raw_times.extend([df['Time'].min(), df['Time'].max()])

    min_bound = min(raw_times) if raw_times else 0
    max_bound = max(raw_times) if raw_times else 100
    max_duration = max_bound - min_bound
    
    # Time shift sliders
    adj_rows = []
    for i in range(0, len(batches), 2):
        row_cols = []
        for j in range(2):
            if i + j < len(batches):
                batch_name = batches[i+j]
                offset = time_offsets.get(batch_name, 0.0)
                control = dbc.Col([
                    html.Strong(batch_name),
                    dbc.Row([
                        dbc.Col(dcc.Slider(id={'type': 'time-shift-slider', 'index': batch_name}, min=-max_duration, max=max_duration, value=offset, updatemode='drag'), width=8),
                        dbc.Col(dcc.Input(id={'type': 'time-shift-num', 'index': batch_name}, type='number', value=offset, className="form-control"), width=4)
                    ])
                ])
                row_cols.append(control)
        adj_rows.append(dbc.Row(row_cols, className="mb-2"))
    
    graph_adjustment_panel = dbc.Card([
        dbc.CardHeader("개별 배치 시간 이동 및 정밀 조정"),
        dbc.CardBody(adj_rows)
    ], className="mb-2")

    # Analysis range controls
    start_range = analysis_range.get('start', min_bound)
    end_range = analysis_range.get('end', max_bound)
    marks = {int(i): str(int(i)) for i in np.linspace(min_bound, max_bound, 10)}

    analysis_range_controls = html.Div([
        dbc.Row([
            dbc.Col(dcc.Input(id='analysis-start-num', type='number', value=start_range), width=2),
            dbc.Col(dcc.RangeSlider(id='analysis-range-slider', min=min_bound, max=max_bound, value=[start_range, end_range], marks=marks, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), width=8),
            dbc.Col(dcc.Input(id='analysis-end-num', type='number', value=end_range), width=2),
        ])
    ], className="mb-2")
    
    return graph_adjustment_panel, analysis_range_controls

# This callback updates the graph. It recalculates the TRUE range on every run.
@app.callback(
    Output('main-graph-tab1', 'figure'),
    [Input('store-batch-data', 'data'),
     Input('store-pv-name', 'data'),
     Input('store-time-offsets', 'data'),
     Input('store-analysis-range', 'data')]
)
def update_main_graph(batch_data, pv_name, time_offsets, analysis_range):
    if not batch_data:
        return go.Figure()

    fig = FigureResampler(go.Figure())
    
    # Calculate the TRUE dynamic range here, including offsets
    all_shifted_times = []
    for name, df_json in batch_data.items():
        df = pd.read_json(df_json, orient='split')
        offset = time_offsets.get(name, 0.0)
        shifted_time = df['Time'] + offset
        fig.add_trace(go.Scattergl(name=name, mode='lines'), hf_x=shifted_time, hf_y=df['Value'])
        if not shifted_time.empty:
            all_shifted_times.extend([shifted_time.min(), shifted_time.max()])

    min_bound = min(all_shifted_times) if all_shifted_times else 0
    max_bound = max(all_shifted_times) if all_shifted_times else 100

    start_range = analysis_range.get('start', min_bound)
    end_range = analysis_range.get('end', max_bound)

    if start_range is not None:
        fig.add_vline(x=start_range, line_width=2, line_dash="dash", line_color="darkviolet", annotation_text="분석 시작")
    if end_range is not None:
        fig.add_vline(x=end_range, line_width=2, line_dash="dash", line_color="darkviolet", annotation_text="분석 종료")

    fig.update_layout(title_text='', xaxis_title='Time (Shifted)', yaxis_title=pv_name, xaxis=dict(range=[min_bound, max_bound]))
    return fig

# This single callback handles ALL interactive controls and updates the stores.
@app.callback(
    [Output('store-time-offsets', 'data'),
     Output('store-analysis-range', 'data')],
    [Input({'type': 'time-shift-slider', 'index': ALL}, 'value'),
     Input({'type': 'time-shift-num', 'index': ALL}, 'value'),
     Input('analysis-range-slider', 'value'),
     Input('analysis-start-num', 'value'),
     Input('analysis-end-num', 'value')],
    [State('store-time-offsets', 'data'),
     State('store-analysis-range', 'data')],
     prevent_initial_call=True
)
def update_stores_from_controls(
    time_slider_vals, time_num_vals, 
    range_slider_val, range_start_num, range_end_num,
    current_offsets, current_range):
    
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    triggered_value = dash.callback_context.triggered[0]['value']
    
    if triggered_value is None:
        return no_update, no_update

    new_offsets = current_offsets.copy()
    new_range = current_range.copy()

    try:
        id_dict = json.loads(triggered_id.split('.')[0])
        if id_dict['type'] == 'time-shift-slider' or id_dict['type'] == 'time-shift-num':
            batch_name = id_dict['index']
            new_offsets[str(batch_name)] = triggered_value
            return new_offsets, no_update
    except (json.JSONDecodeError, KeyError):
        # Not a time-shift control, so must be a range control
        if 'analysis-range-slider' in triggered_id:
            new_range['start'], new_range['end'] = triggered_value
        elif 'analysis-start-num' in triggered_id:
            new_range['start'] = triggered_value
        elif 'analysis-end-num' in triggered_id:
            new_range['end'] = triggered_value
        
        # Prevent range from crossing
        if new_range.get('start') is not None and new_range.get('end') is not None and new_range['start'] > new_range['end']:
            return no_update, no_update
            
        return no_update, new_range
        
    return no_update, no_update

# --- DELETE the old, problematic render callbacks ---
# - render_golden_batch_content
# - render_pls_analysis_content

# --- ADD new callbacks for visibility and dynamic properties ---

# Controls visibility of Golden Batch Tab content
@app.callback(
    [Output('golden-batch-controls-wrapper', 'style'),
     Output('golden-batch-alert-wrapper', 'children')],
    [Input('store-batch-data', 'data'),
     Input('store-quality-data', 'data')]
)
def control_golden_batch_visibility(batch_data, quality_data):
    if batch_data and quality_data:
        return {'display': 'block'}, None
    else:
        alert = dbc.Alert("Step 1에서 배치 데이터와 품질 데이터를 모두 입력해야 합니다.", color="warning", className="m-3")
        return {'display': 'none'}, alert

# Populates the quality target dropdown
@app.callback(
    [Output('dropdown-quality-target', 'options'),
     Output('dropdown-quality-target', 'value')],
    Input('store-quality-data', 'data')
)
def update_quality_target_dropdown(quality_data_json):
    if not quality_data_json:
        return [], None
    quality_df = pd.read_json(quality_data_json, orient='split')
    quality_targets = quality_df.columns[1:].tolist()
    return quality_targets, quality_targets[0]

# Refactor the update_golden_batch_selection callback
@app.callback(
    [Output('checklist-golden-batches', 'options'),
     Output('checklist-golden-batches', 'value')],
    [Input('dropdown-quality-target', 'value'),
     Input('radio-optimization-direction', 'value'),
     Input('store-quality-data', 'data')]
)
def update_golden_batch_selection(target, direction, quality_data_json):
    if not target or not quality_data_json:
        return [], []

    quality_df = pd.read_json(quality_data_json, orient='split')
    batch_name_col = quality_df.columns[0]
    
    is_ascending = direction == 'min'
    sorted_df = quality_df.sort_values(by=target, ascending=is_ascending)
    top_5_batches = sorted_df.head(5)[batch_name_col].tolist()

    all_batches = quality_df[batch_name_col].tolist()
    options = [{'label': b, 'value': b} for b in all_batches]

    return options, top_5_batches

# Controls visibility of PLS Analysis Tab content
@app.callback(
    [Output('pls-analysis-controls-wrapper', 'style'),
     Output('pls-analysis-alert-wrapper', 'children')],
    Input('store-golden-batch-analysis', 'data')
)
def control_pls_visibility(golden_batch_data):
    if golden_batch_data:
        return {'display': 'block'}, None
    else:
        alert = dbc.Alert("먼저 '골든 배치 분석'을 완료해야 합니다.", color="warning", className="m-3")
        return {'display': 'none'}, alert

def align_batch_data(selected_batches, batch_data, time_offsets, analysis_range):
    """
    1. Applies time shifts.
    2. Slices data by analysis range.
    3. Interpolates to a common time index.
    """
    all_dfs = []
    for batch in selected_batches:
        if batch in batch_data:
            df = pd.read_json(batch_data[batch], orient='split')
            offset = time_offsets.get(batch, 0.0)
            df['Time'] = df['Time'] + offset
            
            start_range = analysis_range.get('start')
            end_range = analysis_range.get('end')

            if start_range is not None and end_range is not None:
                df = df[(df['Time'] >= start_range) & (df['Time'] <= end_range)]
            
            all_dfs.append(df.set_index('Time')['Value'].rename(batch))
    
    if not all_dfs:
        return pd.DataFrame(), pd.Index([])

    combined_df = pd.concat(all_dfs, axis=1)
    common_time_index = pd.to_numeric(np.linspace(combined_df.index.min(), combined_df.index.max(), 200)) # Resample to 200 points
    
    aligned_df = combined_df.reindex(combined_df.index.union(common_time_index)).interpolate(method='linear').loc[common_time_index]
    return aligned_df, common_time_index

@app.callback(
    Output('store-golden-batch-analysis', 'data'),
    Input('button-generate-golden-profile', 'n_clicks'),
    [State('checklist-golden-batches', 'value'),
     State('store-batch-data', 'data'),
     State('store-time-offsets', 'data'),
     State('store-analysis-range', 'data')],
    prevent_initial_call=True
)
def generate_golden_profile_data(n_clicks, selected_batches, batch_data, time_offsets, analysis_range):
    if not selected_batches:
        return dash.no_update

    aligned_df, common_time_index = align_batch_data(selected_batches, batch_data, time_offsets, analysis_range)
    
    golden_profile = aligned_df.mean(axis=1)
    std_dev = aligned_df.std(axis=1)

    analysis_results = {
        'aligned_df': aligned_df.to_json(orient='split'),
        'golden_profile': golden_profile.to_json(orient='split'),
        'std_dev': std_dev.to_json(orient='split'),
        'common_time_index': list(common_time_index)
    }
    return analysis_results


@app.callback(
    Output('golden-batch-graph-container', 'children'),
    [Input('store-golden-batch-analysis', 'data'),
     Input('slider-std-dev', 'value')]
)
def update_golden_batch_graph(analysis_data, n_std):
    if not analysis_data:
        return dbc.Alert("위의 '골든 프로파일 생성' 버튼을 클릭하여 분석을 시작하세요.", color='info')

    aligned_df = pd.read_json(analysis_data['aligned_df'], orient='split')
    golden_profile = pd.read_json(analysis_data['golden_profile'], orient='split', typ='series')
    std_dev = pd.read_json(analysis_data['std_dev'], orient='split', typ='series')
    
    fig = go.Figure()

    # Add individual selected batches
    for col in aligned_df.columns:
        fig.add_trace(go.Scatter(x=aligned_df.index, y=aligned_df[col], mode='lines', line=dict(width=1), name=col))

    # Add Golden Corridor
    upper_bound = golden_profile + (n_std * std_dev)
    lower_bound = golden_profile - (n_std * std_dev)
    fig.add_trace(go.Scatter(x=golden_profile.index, y=upper_bound, mode='lines', line=dict(width=0), name=f'Upper Bound (Mean + {n_std}σ)', showlegend=False))
    fig.add_trace(go.Scatter(x=golden_profile.index, y=lower_bound, mode='lines', line=dict(width=0), name=f'Lower Bound (Mean - {n_std}σ)', showlegend=False, fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)'))
    
    # Add Golden Profile
    fig.add_trace(go.Scatter(x=golden_profile.index, y=golden_profile, mode='lines', line=dict(color='orange', width=4), name='Golden Profile (Mean)'))

    fig.update_layout(title="골든 배치 프로파일", xaxis_title="정렬된 시간 (Aligned Time)", yaxis_title="프로세스 변수 값")
    return dcc.Graph(figure=fig)

# ==============================================================================
# 6. Callbacks for Tab 3 (PLS Analysis)
# ==============================================================================
# Modify the triggers for the Tab 3 content callback
@app.callback(
    Output('pls-graph-container', 'children'),
    Input('store-pls-model', 'data')
)
def update_pls_graph(serialized_results):
    if not serialized_results:
        return dbc.Alert("PLS 분석을 실행해주세요.", color="info")

    # Deserialize model/results
    pls_results = pickle.loads(base64.b64decode(serialized_results))
    coef = pls_results['coef'].flatten() # Flatten for line plot
    time_index = pls_results['common_time_index']

    # Create regression coefficient plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=coef, mode='lines', name='회귀 계수'))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    
    fig.update_layout(title='PLS 회귀 계수 (Regression Coefficient)',
                      xaxis_title='정렬된 시간 (Aligned Time)',
                      yaxis_title='계수 값 (Coefficient Value)')

    return dcc.Graph(figure=fig)

@app.callback(
    [Output('store-optimized-profile', 'data'),
     Output('store-debug-log', 'data', allow_duplicate=True)], # Add log output
    Input('button-suggest-profile', 'n_clicks'),
    [State('store-pls-model', 'data'),
     State('radio-optimization-direction', 'value'),
     State('store-golden-batch-analysis', 'data'),
     State('store-debug-log', 'data')],
    prevent_initial_call=True
)
def suggest_optimized_profile(n_clicks, serialized_results, direction, golden_batch_data, current_log):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    new_log = current_log + [f"[{timestamp}] OPT: '최적 프로파일 제안' 버튼 클릭됨."]

    if not serialized_results or not golden_batch_data:
        msg = "최적 프로파일 계산에 필요한 데이터가 부족합니다 (PLS 모델 또는 골든 배치)."
        new_log.append(f"[{timestamp}] OPT ERROR: {msg}")
        return no_update, new_log

    try:
        pls_results = pickle.loads(base64.b64decode(serialized_results))
        golden_profile = pd.read_json(golden_batch_data['golden_profile'], orient='split', typ='series')
        new_log.append(f"[{timestamp}] OPT: 데이터 언패킹 완료.")

        coef = pls_results['coef'].flatten()
        direction_multiplier = 1 if direction == 'max' else -1
        
        # Define a meaningful adjustment step (e.g., 5% of the process standard deviation)
        adjustment_step = golden_profile.std() * 0.05
        new_log.append(f"[{timestamp}] OPT: 조정 단계 계산 완료 (step={adjustment_step:.4f}).")

        # Calculate adjustment based on the SIGN of the coefficient
        adjustment = np.sign(coef) * direction_multiplier * adjustment_step
        
        # Create the new profile using an ADDITIVE method
        optimized_profile = golden_profile + adjustment
        new_log.append(f"[{timestamp}] OPT: 최적 프로파일 계산 완료.")
        
        return optimized_profile.to_json(orient='split'), new_log

    except Exception as e:
        msg = f"최적 프로파일 계산 중 예외 발생: {e}"
        new_log.append(f"[{timestamp}] OPT FATAL ERROR: {msg}")
        return no_update, new_log

@app.callback(
    Output('golden-batch-graph-container', 'children', allow_duplicate=True),
    Input('store-optimized-profile', 'data'),
    [State('golden-batch-graph-container', 'children')], # Get the existing graph
    prevent_initial_call=True
)
def add_optimized_profile_to_graph(optimized_profile_json, existing_graph_children):
    if not optimized_profile_json or not existing_graph_children:
        return dash.no_update

    # Extract the figure from the existing dcc.Graph component
    existing_fig = go.Figure(existing_graph_children['props']['figure'])
    
    optimized_profile = pd.read_json(optimized_profile_json, orient='split', typ='series')
    
    # Add the new trace for the optimized profile
    existing_fig.add_trace(
        go.Scatter(x=optimized_profile.index, y=optimized_profile, mode='lines', 
                   line=dict(color='red', width=4, dash='dash'), 
                   name='Optimized Profile')
    )
    
    return dcc.Graph(figure=existing_fig)

@app.callback(
    Output('div-quality-data-table-container', 'children'), # <-- Changed Output ID
    Input('store-quality-data', 'data')
)
def update_quality_data_table(quality_data_json):
    if quality_data_json is None:
        return []

    try:
        df = pd.read_json(quality_data_json, orient='split')
        table = dash_table.DataTable(
            id='datatable-quality',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            editable=True,
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
        return table # Only return the table
    except Exception as e:
        return [dbc.Alert(f"품질 데이터 표시 오류: {e}", color="danger")]

# --- Add a NEW callback to control the visibility of the buttons ---
@app.callback(
    [Output('button-save-quality-changes', 'style'),
     Output('button-reset-quality', 'style')],
    Input('store-quality-data', 'data')
)
def control_quality_buttons_visibility(quality_data):
    if quality_data:
        visible_style = {'display': 'inline-block'}
        return visible_style, visible_style
    else:
        hidden_style = {'display': 'none'}
        return hidden_style, hidden_style

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, dbc.Alert("CSV 또는 XLSX 파일만 지원됩니다.", color="warning")
    except Exception as e:
        return None, dbc.Alert(f"파일 파싱 오류: {e}", color="danger")
    return df, None

@app.callback(
    [Output('store-quality-data', 'data', allow_duplicate=True),
     Output('output-batch-add-status', 'children', allow_duplicate=True),
     Output('upload-quality-data', 'contents'),
     Output('store-debug-log', 'data', allow_duplicate=True)],
    Input('upload-quality-data', 'contents'),
    [State('upload-quality-data', 'filename'),
     State('store-debug-log', 'data')],
    prevent_initial_call=True
)
def update_quality_data_store(contents, filename, current_log):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if contents is None:
        return no_update, no_update, no_update, no_update
    
    df, error_alert = parse_contents(contents, filename)
    if error_alert:
        new_log = current_log + [f"[{timestamp}] UPLOAD: 파일 처리 오류."]
        return no_update, error_alert, None, new_log

    alert = dbc.Alert("품질 데이터 업로드 완료.", color="success", dismissable=True)
    new_log = current_log + [f"[{timestamp}] UPLOAD: 파일 데이터를 저장소에 썼습니다."]
    return df.to_json(orient='split'), alert, None, new_log

@app.callback(
    [Output('store-quality-data', 'data', allow_duplicate=True),
     Output('debug-save-action', 'children'),
     Output('store-debug-log', 'data', allow_duplicate=True)],
    Input('button-save-quality-changes', 'n_clicks'),
    [State('datatable-quality', 'data'),
     State('store-debug-log', 'data')],
    prevent_initial_call=True
)
def save_quality_changes(n_clicks, table_rows, current_log):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if not table_rows:
        debug_msg = f"[{timestamp}] SAVE: 테이블 데이터를 받지 못함."
        new_log = current_log + [debug_msg]
        return no_update, debug_msg, new_log

    df = pd.DataFrame(table_rows)
    debug_msg = f"[{timestamp}] SAVE: {len(df)}개 행을 저장소에 썼습니다."
    new_log = current_log + [debug_msg]
    return df.to_json(orient='split'), debug_msg, new_log

@app.callback(
    [Output('store-quality-data', 'data', allow_duplicate=True),
     Output('store-debug-log', 'data', allow_duplicate=True)],
    Input('button-reset-quality', 'n_clicks'),
    [State('store-debug-log', 'data')],
    prevent_initial_call=True
)
def reset_quality_data(n_clicks, current_log):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    new_log = current_log + [f"[{timestamp}] RESET: 저장소를 비웠습니다."]
    return None, new_log

@app.callback(
    Output('debug-output', 'children'),
    [Input('store-batch-data', 'data'),
     Input('store-quality-data', 'data'),
     Input('store-debug-log', 'data')] # <-- Add this as Input
)
def debug_output(batch_data, quality_data, debug_log):
    batch_status = "✅ [배치 데이터 있음]" if batch_data else "❌ [배치 데이터 없음]"
    quality_status = "✅ [품질 데이터 있음]" if quality_data else "❌ [품질 데이터 없음]"
    
    log_str = "\n".join(debug_log)
    
    return f"배치 데이터 저장소 상태: {batch_status}\n품질 데이터 저장소 상태: {quality_status}\n\n--- 이벤트 로그 ---\n{log_str}"

# Modify the run_pls_analysis callback to add detailed logging

@app.callback(
    [Output('store-pls-model', 'data'),
     Output('pls-status-output', 'children'),
     Output('store-debug-log', 'data', allow_duplicate=True)], # Add log output
    Input('button-run-pls', 'n_clicks'),
    [State('input-n-components', 'value'),
     State('store-golden-batch-analysis', 'data'),
     State('store-quality-data', 'data'),
     State('dropdown-quality-target', 'value'),
     State('store-debug-log', 'data')], # Add log state
    prevent_initial_call=True
)
def run_pls_analysis(n_clicks, n_components, golden_batch_data, quality_data_json, quality_target, current_log):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    new_log = current_log + [f"[{timestamp}] PLS: '분석 실행' 버튼 클릭됨."]

    if not all([golden_batch_data, quality_data_json, quality_target]):
        msg = "PLS 분석에 필요한 데이터가 부족합니다 (골든 배치, 품질 데이터, 품질 목표)."
        new_log.append(f"[{timestamp}] PLS ERROR: {msg}")
        return no_update, dbc.Alert(msg, color="warning"), new_log

    try:
        aligned_df = pd.read_json(golden_batch_data['aligned_df'], orient='split')
        quality_df = pd.read_json(quality_data_json, orient='split')
        batch_name_col = quality_df.columns[0]
        new_log.append(f"[{timestamp}] PLS: 데이터 언패킹 완료.")

        X = aligned_df.transpose()
        quality_df = quality_df.set_index(batch_name_col)
        Y = quality_df.loc[X.index, quality_target]
        new_log.append(f"[{timestamp}] PLS: 데이터 준비 완료. X shape={X.shape}, Y shape={Y.shape}")

        if Y.std() == 0:
            msg = "분석 오류: 선택된 배치들의 품질 값이 모두 동일하여 PLS 분석을 수행할 수 없습니다."
            new_log.append(f"[{timestamp}] PLS ERROR: {msg}")
            return no_update, dbc.Alert(msg, color='danger'), new_log

        new_log.append(f"[{timestamp}] PLS: 모델 학습 시작 (n_components={n_components})...")
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(X.values, Y.values)
        new_log.append(f"[{timestamp}] PLS: 모델 학습 완료.")

        pls_results = {
            'coef': pls.coef_,
            'common_time_index': golden_batch_data['common_time_index'],
            'x_scores': pls.x_scores_.tolist()
        }
        serialized_results = base64.b64encode(pickle.dumps(pls_results)).decode('utf-8')
        new_log.append(f"[{timestamp}] PLS: 분석 결과 직렬화 완료.")
        
        return serialized_results, dbc.Alert("PLS 분석 완료!", color="success", dismissable=True), new_log

    except Exception as e:
        msg = f"PLS 분석 중 예외 발생: {e}"
        new_log.append(f"[{timestamp}] PLS FATAL ERROR: {msg}")
        return no_update, dbc.Alert(msg, color="danger"), new_log

if __name__ == '__main__':
    app.run(debug=True) 