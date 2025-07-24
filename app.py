import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, unregister_plotly_resampler
unregister_plotly_resampler()
from sklearn.cross_decomposition import PLSRegression

st.set_page_config(layout="wide")

# --- Session State Initialization (Correct and Safe) ---
if 'batches' not in st.session_state:
    st.session_state.batches = []
if 'quality_data' not in st.session_state:
    st.session_state.quality_data = None
if 'pv_name' not in st.session_state:
    st.session_state.pv_name = "Value"
if 'time_offsets' not in st.session_state:
    st.session_state.time_offsets = {}
if 'analysis_start_time' not in st.session_state:
    st.session_state.analysis_start_time = None
if 'analysis_end_time' not in st.session_state:
    st.session_state.analysis_end_time = None

st.title("배치 데이터 분석 플랫폼")

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📊 데이터 시각화 및 정렬",
    "🌟 골든 배치 분석",
    "📈 PLS 원인 분석"
])

# ==============================================================================
# Tab 1: Data Visualization and Alignment
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 2])

    # --- Left Column: Data Input ---
    with col1:
        st.header("Step 1: 배치 데이터 입력")

        with st.form("batch_form", clear_on_submit=True):
            st.write("분석할 배치 데이터를 하나씩 추가하세요.")
            time_unit_options = {
                "1 minute": 1, "5 minutes": 5, "10 minutes": 10,
                "30 minutes": 30, "1 hour": 60,
            }
            selected_time_unit = st.selectbox("시간 간격 선택:", options=list(time_unit_options.keys()))
            batch_name = st.text_input("배치명:", placeholder="예: IBC24001")
            pv_name_input = st.text_input("프로세스 변수(PV) 이름:", value=st.session_state.pv_name)
            data_paste_area = st.text_area("데이터 붙여넣기:", height=200, placeholder="엑셀의 한 열을 복사하여 여기에 붙여넣으세요...")

            submitted = st.form_submit_button("배치 추가")
            if submitted:
                if not batch_name or not data_paste_area:
                    st.warning("배치명과 데이터를 모두 입력해주세요.")
                else:
                    try:
                        lines = data_paste_area.strip().split('\n')
                        values = [float(line.strip()) for line in lines]
                        interval = time_unit_options[selected_time_unit]
                        time_index = [i * interval for i in range(len(values))]
                        df = pd.DataFrame({'Time': time_index, 'Value': values})

                        st.session_state.batches.append({'name': batch_name, 'data': df})
                        st.session_state.pv_name = pv_name_input
                        st.session_state.time_offsets[batch_name] = 0.0
                        st.rerun()
                    except ValueError:
                        st.error("데이터에 숫자가 아닌 값이 포함되어 있습니다. 숫자 데이터만 입력해주세요.")
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")

        if st.session_state.batches:
            st.subheader("추가된 배치 목록")
            for i in range(len(st.session_state.batches) - 1, -1, -1):
                batch = st.session_state.batches[i]
                b_col1, b_col2 = st.columns([4, 1])
                with b_col1:
                    st.write(f"**{batch['name']}** ({len(batch['data'])} points)")
                with b_col2:
                    if st.button("삭제", key=f"del_batch_{i}", use_container_width=True):
                        batch_to_delete = st.session_state.batches.pop(i)
                        if batch_to_delete['name'] in st.session_state.time_offsets:
                            del st.session_state.time_offsets[batch_to_delete['name']]
                        st.rerun()

            st.divider()
            if st.button("모든 배치 초기화", type="secondary"):
                st.session_state.batches = []
                st.session_state.time_offsets = {}
                st.session_state.analysis_start_time = None
                st.session_state.analysis_end_time = None
                st.rerun()

        st.divider()

        st.header("Step 2: 품질 데이터 업로드")
        uploaded_quality_file = st.file_uploader("품질 데이터 파일 업로드 (CSV/XLSX)", type=["csv", "xlsx"], key="quality_uploader")
        if uploaded_quality_file:
            try:
                if st.session_state.quality_data is None:
                     pass # clear_pls_results() # Removed as per edit hint
                df = pd.read_csv(uploaded_quality_file) if uploaded_quality_file.name.endswith('.csv') else pd.read_excel(uploaded_quality_file)
                st.session_state.quality_data = df
                st.success("품질 데이터를 성공적으로 업로드했습니다.")
            except Exception as e:
                st.error(f"품질 데이터 파일을 읽는 중 오류 발생: {e}")
                st.session_state.quality_data = None

        if st.session_state.quality_data is not None:
            st.info("아래 표에서 직접 품질 데이터를 수정할 수 있습니다.")
            st.session_state.quality_data = st.data_editor(st.session_state.quality_data, use_container_width=True, height=300)
            if st.button("품질 데이터 초기화", type="secondary", key="clear_quality_data"):
                st.session_state.quality_data = None
                st.rerun()

    # --- Right Column: Visualization ---
    with col2:
        st.header("배치 프로파일")

        if not st.session_state.batches:
            st.info("좌측 메뉴에서 배치 데이터를 추가하면 여기에 그래프가 표시됩니다.")
        else:
            # 1. Individual Time Shift Controls
            st.subheader("그래프 조정")
            with st.expander("개별 배치 시간 이동 및 정밀 조정 (컴팩트 뷰)"):
                max_time = max((b['data']['Time'].max() for b in st.session_state.batches), default=100)
                batches = st.session_state.batches
                for i in range(0, len(batches), 2):
                    row_cols = st.columns(2)
                    with row_cols[0]:
                        batch_left = batches[i]
                        batch_name_left = batch_left['name']
                        current_offset_left = st.session_state.time_offsets.get(batch_name_left, 0.0)
                        
                        st.markdown(f"**{batch_name_left}**")
                        
                        # Create nested columns for slider and number input
                        sub_cols_left = st.columns([3, 1])
                        with sub_cols_left[0]:
                            slider_val_left = st.slider(f"'{batch_name_left}' 이동", -float(max_time), float(max_time), float(current_offset_left), key=f"slider_{batch_name_left}", label_visibility="collapsed")
                        with sub_cols_left[1]:
                            num_val_left = st.number_input("정밀 조정 값", -float(max_time), float(max_time), float(current_offset_left), step=1.0, key=f"num_{batch_name_left}", label_visibility="collapsed")
                        
                        if slider_val_left != current_offset_left:
                            st.session_state.time_offsets[batch_name_left] = slider_val_left
                            st.rerun()
                        elif num_val_left != current_offset_left:
                            st.session_state.time_offsets[batch_name_left] = num_val_left
                            st.rerun()

                    # --- Right Batch in the Row (if it exists) ---
                    if i + 1 < len(batches):
                        with row_cols[1]:
                            batch_right = batches[i+1]
                            batch_name_right = batch_right['name']
                            current_offset_right = st.session_state.time_offsets.get(batch_name_right, 0.0)
                            
                            st.markdown(f"**{batch_name_right}**")

                            # Create nested columns for slider and number input
                            sub_cols_right = st.columns([3, 1])
                            with sub_cols_right[0]:
                                slider_val_right = st.slider(f"'{batch_name_right}' 이동", -float(max_time), float(max_time), float(current_offset_right), key=f"slider_{batch_name_right}", label_visibility="collapsed")
                            with sub_cols_right[1]:
                                num_val_right = st.number_input("정밀 조정 값", -float(max_time), float(max_time), float(current_offset_right), step=1.0, key=f"num_{batch_name_right}", label_visibility="collapsed")
                            
                            if slider_val_right != current_offset_right:
                                st.session_state.time_offsets[batch_name_right] = slider_val_right
                                st.rerun()
                            elif num_val_right != current_offset_right:
                                st.session_state.time_offsets[batch_name_right] = num_val_right
                                st.rerun()
            
            # 2. Integrated Analysis Range Controls
            all_shifted_times = []
            for b in st.session_state.batches:
                offset = st.session_state.time_offsets.get(b['name'], 0.0)
                all_shifted_times.append(b['data']['Time'].min() + offset)
                all_shifted_times.append(b['data']['Time'].max() + offset)
            
            min_bound = min(all_shifted_times) if all_shifted_times else 0.0
            max_bound = max(all_shifted_times) if all_shifted_times else 100.0

            if st.session_state.get('analysis_start_time') is None or st.session_state.get('analysis_end_time') is None:
                st.session_state.analysis_start_time = min_bound
                st.session_state.analysis_end_time = max_bound

            current_start = st.session_state.analysis_start_time
            current_end = st.session_state.analysis_end_time

            # --- Create three columns for the compact layout ---
            col_start, col_slider, col_end = st.columns([1, 8, 1]) # Adjusted ratio for smaller number boxes

            with col_start:
                num_start = st.number_input("분석 시작 시간", value=float(current_start), step=1.0, label_visibility="collapsed")

            with col_slider:
                slider_start, slider_end = st.slider(
                    "분석 구간 조절",
                    min_value=float(min_bound),
                    max_value=float(max_bound),
                    value=(float(current_start), float(current_end)),
                    label_visibility="collapsed"
                )

            with col_end:
                num_end = st.number_input("분석 종료 시간", value=float(current_end), step=1.0, label_visibility="collapsed")
            
            # --- Sync Logic ---
            if (num_start, num_end) != (current_start, current_end):
                st.session_state.analysis_start_time = num_start
                st.session_state.analysis_end_time = num_end
                st.rerun()
            elif (slider_start, slider_end) != (current_start, current_end):
                st.session_state.analysis_start_time = slider_start
                st.session_state.analysis_end_time = slider_end
                st.rerun()

            # The divider is removed as per the previous request
            # st.divider()
            
            # 3. The Graph Body
            fig = FigureResampler(go.Figure())
            for batch in st.session_state.batches:
                batch_name = batch['name']
                df = batch['data'].copy()
                offset = st.session_state.time_offsets.get(batch_name, 0.0)
                x_data = df['Time'] + offset
                y_data = df['Value']
                fig.add_trace(go.Scattergl(name=batch_name, mode='lines'), hf_x=x_data, hf_y=y_data)
            
            if st.session_state.get('analysis_start_time') is not None and st.session_state.get('analysis_end_time') is not None:
                fig.add_vline(x=st.session_state.analysis_start_time, line_width=2, line_dash="dash", line_color="darkviolet", annotation_text="분석 시작", annotation_position="top right")
                fig.add_vline(x=st.session_state.analysis_end_time, line_width=2, line_dash="dash", line_color="darkviolet", annotation_text="분석 종료", annotation_position="top left")

            fig.update_layout(
                title_text='',
                xaxis_title='Time (Shifted)',
                yaxis_title=st.session_state.pv_name,
                xaxis=dict(rangeslider=dict(visible=False), type="linear")
            )
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Tab 2: Golden Batch Analysis
# ==============================================================================
with tab2:
    st.header("동적 골든 배치 분석")

    if st.session_state.quality_data is None:
        st.warning("먼저 '데이터 시각화 및 정렬' 탭에서 품질 데이터를 업로드해주세요.")
    else:
        st.subheader("Step 1: 분석 목표 설정")

        quality_df = st.session_state.quality_data

        # Select only numeric columns for analysis
        numeric_cols = quality_df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.error("품질 데이터에 분석할 수 있는 숫자형 데이터가 없습니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "분석할 품질 목표를 선택하세요:",
                    numeric_cols,
                    key="g_target_col"
                )
            with col2:
                st.radio(
                    "최적화 방향을 선택하세요:",
                    ("최대화", "최소화"),
                    horizontal=True,
                    key="g_optimization_direction",
                )

            st.divider()

            st.subheader("Step 2: 골든 배치 선택")

            # Find the batch ID column
            all_cols = quality_df.columns.tolist()
            default_batch_col = all_cols[0]
            for col in all_cols:
                if 'batch' in col.lower() or '배치' in col.lower():
                    default_batch_col = col
                    break

            # Sort quality data based on target and direction
            sorted_df = quality_df.sort_values(
                by=st.session_state.g_target_col,
                ascending=(st.session_state.g_optimization_direction == "최소화")
            )

            # Get available batch names from session state
            available_batches = [b['name'] for b in st.session_state.batches]

            # Filter sorted_df to only include available batches
            sorted_df = sorted_df[sorted_df[default_batch_col].isin(available_batches)]

            # Recommend top 5 batches
            top_5_batches = sorted_df[default_batch_col].head(5).tolist()

            st.multiselect(
                "분석에 사용할 배치를 선택하세요. (상위 5개가 자동으로 추천됩니다)",
                options=available_batches,
                default=top_5_batches,
                key="g_selected_batches",
            )

            # --- Data Alignment and Interpolation ---
            if st.session_state.g_selected_batches:
                aligned_data = []

                analysis_start = st.session_state.get('analysis_start_time')
                analysis_end = st.session_state.get('analysis_end_time')

                if analysis_start is None or analysis_end is None:
                    st.warning("분석 구간이 설정되지 않았습니다. '데이터 시각화 및 정렬' 탭에서 구간을 설정해주세요.")
                elif analysis_start >= analysis_end:
                    st.error("분석 시작 시간은 종료 시간보다 빨라야 합니다.")
                else:
                    selected_batch_objects = [b for b in st.session_state.batches if b['name'] in st.session_state.g_selected_batches]

                    st.info(f"지정된 분석 구간: `{analysis_start:.2f}` ~ `{analysis_end:.2f}`")

                    # Resample data in the user-defined time range
                    num_points = 500
                    resampled_time = np.linspace(analysis_start, analysis_end, num_points)

                    for batch in selected_batch_objects:
                        offset = st.session_state.time_offsets.get(batch['name'], 0.0)
                        original_time = batch['data']['Time'] + offset
                        original_value = batch['data']['Value']

                        interpolated_value = np.interp(resampled_time, original_time, original_value)

                        aligned_data.append({
                            'name': batch['name'],
                            'time': resampled_time,
                            'value': interpolated_value
                        })

                    st.session_state.aligned_golden_data = aligned_data

            st.divider()

            st.subheader("Step 3: 골든 프로파일 시각화")

            if 'aligned_golden_data' in st.session_state and st.session_state.aligned_golden_data:

                # --- Calculate Golden Profile and Corridor ---
                aligned_values = np.array([d['value'] for d in st.session_state.aligned_golden_data])
                mean_profile = np.mean(aligned_values, axis=0)
                std_profile = np.std(aligned_values, axis=0)

                time_axis = st.session_state.aligned_golden_data[0]['time']

                n_std = st.slider("허용 범위(표준편차 계수 N)를 조절하세요:", 1.0, 4.0, 2.0, 0.5)

                upper_bound = mean_profile + (n_std * std_profile)
                lower_bound = mean_profile - (n_std * std_profile)

                # --- Plotting ---
                fig = go.Figure()

                # Add individual selected batches
                for batch_data in st.session_state.aligned_golden_data:
                    fig.add_trace(go.Scatter(
                        x=batch_data['time'],
                        y=batch_data['value'],
                        mode='lines',
                        line=dict(color='lightgrey', width=1),
                        name=batch_data['name'],
                        showlegend=False
                    ))

                # Add corridor
                fig.add_trace(go.Scatter(
                    x=np.concatenate([time_axis, time_axis[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name='허용 범위 (±Nσ)'
                ))

                # Add Golden Profile
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=mean_profile,
                    mode='lines',
                    line=dict(color='green', width=3),
                    name='골든 프로파일 (평균)'
                ))

                # Add Optimized Profile if it exists
                if 'optimized_profile' in st.session_state:
                    opt_profile = st.session_state.optimized_profile
                    fig.add_trace(go.Scatter(
                        x=opt_profile['time'],
                        y=opt_profile['value'],
                        mode='lines',
                        line=dict(color='orange', width=3, dash='dash'),
                        name='최적 제안 프로파일'
                    ))

                fig.update_layout(
                    title="골든 프로파일 및 선택된 배치",
                    xaxis_title="Time (Aligned)",
                    yaxis_title=st.session_state.pv_name,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("분석할 배치를 선택하면 여기에 골든 프로파일이 표시됩니다.")


# ==============================================================================
# Tab 3: PLS Root Cause Analysis
# ==============================================================================
with tab3:
    st.header("PLS 기반 원인 분석")

    if 'aligned_golden_data' not in st.session_state or not st.session_state.aligned_golden_data or len(st.session_state.aligned_golden_data) < 2:
        st.warning("먼저 '골든 배치 분석' 탭에서 분석할 배치를 2개 이상 선택하고, 골든 프로파일을 생성해주세요.")
    else:
        # --- NEW: Fingerprint validation ---
        pls_fingerprint = st.session_state.get('pls_fingerprint', None)
        current_fingerprint = {
            'batches': st.session_state.g_selected_batches,
            'target': st.session_state.g_target_col,
            'range': (st.session_state.analysis_start_time, st.session_state.analysis_end_time)
        }

        # If the settings have changed since the last PLS run, invalidate the results
        if pls_fingerprint and pls_fingerprint != current_fingerprint:
            # Manually clear the keys here instead of calling a function
            for key in ['pls_model', 'pls_X_data', 'pls_y_data', 'optimized_profile', 'pls_fingerprint']:
                if key in st.session_state:
                    del st.session_state[key]
            st.warning("골든 배치 선택, 품질 목표, 또는 분석 구간이 변경되었습니다. PLS 분석을 다시 실행해주세요.")

        st.info(f"현재 PLS 분석은 **'{st.session_state.g_target_col}'** 품질 목표에 대해 수행됩니다.")
        st.subheader("Step 1: PLS 모델 학습")

        # Ensure n_components is less than n_samples
        max_components = len(st.session_state.aligned_golden_data) - 1

        n_components = st.number_input(
            "PLS 모델의 구성요소(Component) 개수를 선택하세요:",
            min_value=1,
            max_value=max_components,
            value=min(2, max_components), # Default to 2 or max_components if smaller
            step=1,
            help="데이터의 복잡성을 얼마나 반영할지 결정합니다. 보통 1~3 사이의 값을 사용합니다."
        )

        if st.button("PLS 분석 실행", key="pls_run_button", type="primary"):
            with st.spinner("PLS 모델을 학습하는 중입니다..."):
                # 1. Prepare data (X and y)
                aligned_data = st.session_state.aligned_golden_data
                X = np.array([d['value'] for d in aligned_data])
                batch_names_in_order = [d['name'] for d in aligned_data]

                quality_df = st.session_state.quality_data

                # Find the batch ID column
                all_cols = quality_df.columns.tolist()
                batch_col = all_cols[0]
                for col in all_cols:
                    if 'batch' in col.lower() or '배치' in col.lower():
                        batch_col = col
                        break

                # Get y values in the correct order
                y_df = quality_df[quality_df[batch_col].isin(batch_names_in_order)]
                y_df_ordered = y_df.set_index(batch_col).loc[batch_names_in_order].reset_index()
                y = y_df_ordered[st.session_state.g_target_col].values

                # --- TEMPORARY DEBUGGING ---
                st.subheader("🕵️‍♂️ 디버깅 정보")
                st.write("아래 정보를 확인하여 품질 값이 실제로 동일한지 확인해주세요.")
                debug_df = pd.DataFrame({
                    "선택된 배치명": batch_names_in_order,
                    f"품질 값 ({st.session_state.g_target_col})": y
                })
                st.dataframe(debug_df)
                st.write(f"배치 ID로 사용된 컬럼: `{batch_col}`")
                # --- END DEBUGGING ---

                # --- NEW: Validate y data before fitting ---
                if np.std(y) < 1e-9:
                    st.error("분석 오류: 선택된 배치들의 품질 값이 모두 동일합니다. PLS 분석을 수행하려면 품질 값이 서로 다른 배치들을 2개 이상 선택해주세요.")
                    # Clear any previous pls_model to avoid showing old results
                    if 'pls_model' in st.session_state:
                        del st.session_state.pls_model
                else:
                    # 2. Train PLS model
                    pls = PLSRegression(n_components=n_components)
                    pls.fit(X, y.reshape(-1, 1))

                    # 3. Store results
                    st.session_state.pls_model = pls
                    st.session_state.pls_X_data = X
                    st.session_state.pls_y_data = y

                    # --- NEW: Store the fingerprint along with the model ---
                    st.session_state.pls_fingerprint = current_fingerprint
                    st.success("PLS 모델 학습이 완료되었습니다!")

        if 'pls_model' in st.session_state:
            st.divider()

            st.subheader("Step 2: 핵심 영향 인자 분석")

            try:
                pls = st.session_state.pls_model

                if not hasattr(pls, 'coef_'):
                    st.error("학습된 PLS 모델에서 회귀 계수를 찾을 수 없습니다.")
                else:
                    coefs = pls.coef_.flatten()

                    if coefs is None or coefs.size == 0:
                        st.error("계산된 회귀 계수가 비어있습니다. 모델 학습을 다시 시도해주세요.")
                    else:
                        time_axis = st.session_state.aligned_golden_data[0]['time']

                        fig = go.Figure()

                        colors = ['red' if c < 0 else 'blue' for c in coefs]
                        fig.add_trace(go.Bar(
                            x=time_axis,
                            y=coefs,
                            marker_color=colors,
                            name='회귀 계수'
                        ))

                        fig.update_layout(
                            title='PLS 회귀 계수 (영향력 분석)',
                            xaxis_title='Time (Aligned)',
                            yaxis_title='계수 값 (영향력)',
                            showlegend=False
                        )

                        st.info("""
                        **그래프 해석:**
                        - **양수(파란색) 막대:** 이 시간대의 공정 변수 값이 높을수록 최종 품질이 향상됩니다.
                        - **음수(빨간색) 막대:** 이 시간대의 공정 변수 값이 높을수록 최종 품질이 저하됩니다.
                        - 막대의 높이는 영향력의 크기를 의미합니다.
                        """)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"그래프를 그리는 중 오류가 발생했습니다: {e}")

            st.divider()

            st.subheader("Step 3: 이론적 최적 프로파일 제안")

            st.info("""
            아래 버튼을 클릭하면 PLS 분석 결과를 바탕으로, 현재의 골든 프로파일을 개선할 수 있는
            **이론적 최적 프로파일**을 계산합니다. 계산된 프로파일은 '골든 배치 분석' 탭의 그래프에
            '최적 제안'이라는 이름으로 추가되어 기존 프로파일과 비교할 수 있습니다.
            """)

            if st.button("최적 프로파일 계산 및 적용", key="calc_optimal_profile"):
                with st.spinner("최적 프로파일을 계산 중입니다..."):

                    # 1. Calculate the optimal direction based on coefficients
                    direction = st.session_state.g_optimization_direction
                    coefs = pls.coef_.flatten()

                    # We move in the direction of the coefficients for maximization, and opposite for minimization
                    # The magnitude is scaled by the std of coefficients to make it reasonable
                    optimal_direction = coefs if direction == "최대화" else -coefs
                    scaled_direction = optimal_direction * np.std(st.session_state.pls_X_data) / np.std(coefs)

                    # --- NEW: Safety check for division by zero ---
                    if np.std(coefs) < 1e-9:
                        st.warning("모델의 영향력 계수가 모두 동일하여 의미 있는 최적 프로파일을 제안할 수 없습니다. 다른 배치 조합이나 더 많은 Component로 시도해보세요.")
                        # Reset optimized_profile if it exists from a previous run
                        if 'optimized_profile' in st.session_state:
                            del st.session_state.optimized_profile
                    else:
                        # 2. Calculate the new optimal profile
                        current_golden_profile = np.mean(st.session_state.pls_X_data, axis=0)
                        optimized_profile = current_golden_profile + scaled_direction

                        # 3. Store it in session state
                        st.session_state.optimized_profile = {
                            'time': time_axis,
                            'value': optimized_profile
                        }
                        st.success("최적 프로파일 계산이 완료되었습니다! '골든 배치 분석' 탭에서 확인하세요.")

            # The rest of Phase 3 will be developed here 