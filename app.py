import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
# from streamlit_plotly_events import plotly_events # 더 이상 사용하지 않음

st.set_page_config(layout="wide")

st.title("Interactive Batch Process Optimization Platform")

# --- 모든 세션 상태를 여기서 한 번에 초기화 ---
if 'page' not in st.session_state:
    st.session_state.page = "1. Data Processing"
    st.session_state.processed_data = None
    st.session_state.quality_data = None
    st.session_state.pv_name = None
    st.session_state.aligned_data = None
    st.session_state.analysis_data = None
    st.session_state.golden_batch_profile = None
    st.session_state.pls_coefficients = None
    st.session_state.optimized_profile = None
    st.session_state.target_quality = None
    st.session_state.optimization_direction = 'Maximize'


# --- 사이드바 네비게이션 ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1. Data Processing",
        "2. Time Alignment",
        "3. Analysis Range",
        "4. Golden Batch",
        "5. PLS & Optimization"
    ],
    key='page'
)

# ==============================================================================
# 1. 데이터 처리 페이지
# ==============================================================================
if page == "1. Data Processing":
    st.sidebar.header("Step 1: Upload & Process Data")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload your CSV or Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # --- 데이터 로딩 및 전처리 UI ---
        pv_name_candidate = uploaded_files[0].name.split('.')[0]
        pv_name = st.sidebar.text_input("Enter Process Variable (PV) name:", value=pv_name_candidate)
        
        # 임시로 첫번째 파일을 읽어 컬럼 목록 생성
        try:
            # Streamlit의 UploadedFile 객체는 파일 포인터를 가질 수 있으므로 seek(0)으로 초기화
            uploaded_files[0].seek(0)
            temp_df = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith('.csv') else pd.read_excel(uploaded_files[0])
            
            if temp_df.empty:
                st.error(f"File '{uploaded_files[0].name}' appears to be empty. Please upload a valid data file.")
                st.stop()

            all_columns = temp_df.columns.tolist()
            non_batch_columns = [col for col in all_columns if col != 'Batch']
            
            # Defensive index calculation
            default_index = max(0, len(non_batch_columns) - 5)
            if not non_batch_columns:
                st.error("No data columns found in the file (excluding 'Batch'). Please check the file format.")
                st.stop()

            first_quality_col = st.sidebar.selectbox(
                "Select the first quality data column:",
                options=non_batch_columns,
                index=default_index
            )

            if st.sidebar.button("Process All Uploaded Data"):
                with st.spinner("Processing data..."):
                    combined_df = pd.DataFrame()
                    for file in uploaded_files:
                        try:
                            file.seek(0) # 각 파일 처리 전 포인터 리셋
                            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                            if 'Batch' in df.columns:
                                df.dropna(subset=['Batch'], inplace=True)
                                if not df.empty:
                                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                        except EmptyDataError:
                            st.warning(f"Skipping empty file: {file.name}")
                        except Exception as e:
                            st.warning(f"Skipping file {file.name} due to an error: {e}")
                    
                    if combined_df.empty:
                        st.error("No valid data could be processed from the uploaded files.")
                        st.stop()

                    # !!! BUG FIX: Batch ID를 문자열로 통일하여 데이터 타입 불일치 오류 방지 !!!
                    combined_df['Batch'] = combined_df['Batch'].astype(str)

                    first_quality_idx = all_columns.index(first_quality_col)
                    quality_cols = all_columns[first_quality_idx:]
                    st.session_state.quality_data = combined_df[['Batch'] + quality_cols]
                    
                    timeseries_cols = all_columns[1:first_quality_idx]
                    melted_df = pd.melt(combined_df, id_vars=['Batch'], value_vars=timeseries_cols, var_name='Time', value_name=pv_name)
                    melted_df['Time'] = melted_df['Time'].str.extract(r'(\d+)').astype(int)
                    
                    st.session_state.processed_data = melted_df
                    st.session_state.pv_name = pv_name
                    st.success("Data processing complete!")
        
        except EmptyDataError:
            st.error(f"Error: The first file '{uploaded_files[0].name}' is empty. The application needs at least one valid file to determine the data structure.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred: {e}. Please ensure your files are correctly formatted.")
            st.stop()

    # --- 메인 화면 출력 ---
    st.header("1. Data Processing Results")
    if st.session_state.processed_data is not None:
        st.subheader("Processed Time-Series Data (Long Format)")
        st.dataframe(st.session_state.processed_data.head())
        
        st.subheader("Quality Data")
        st.dataframe(st.session_state.quality_data.head())

        st.subheader("Initial Batch-wise Plot")
        fig = px.line(st.session_state.processed_data, x='Time', y=st.session_state.pv_name, color='Batch')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload and process data files using the sidebar controls.")


# ==============================================================================
# 2. 시간 정렬 페이지
# ==============================================================================
elif page == "2. Time Alignment":
    if st.session_state.processed_data is None:
        st.warning("Please process data in Step 1 first.")
    else:
        st.sidebar.header("Step 2: Align Time")
        alignment_method = st.sidebar.radio("Select alignment method:", ('Align to Minimum Value', 'Align to Maximum Value'))
        
        if st.sidebar.button("Align Data"):
            with st.spinner("Aligning data..."):
                df = st.session_state.processed_data.copy()
                pv_name = st.session_state.pv_name
                aligned_dfs = []
                for batch_id in df['Batch'].unique():
                    batch_df = df[df['Batch'] == batch_id].copy()
                    if not batch_df[pv_name].dropna().empty:
                        anchor_time = batch_df.loc[batch_df[pv_name].idxmin()]['Time'] if alignment_method == 'Align to Minimum Value' else batch_df.loc[batch_df[pv_name].idxmax()]['Time']
                        batch_df['Aligned_Time'] = batch_df['Time'] - anchor_time
                        aligned_dfs.append(batch_df)
                st.session_state.aligned_data = pd.concat(aligned_dfs, ignore_index=True)
                st.success("Data alignment successful!")

        # --- 메인 화면 출력 ---
        st.header("2. Time Alignment Results")
        if st.session_state.aligned_data is not None:
            st.subheader("Data After Alignment")
            fig = px.line(st.session_state.aligned_data, x='Aligned_Time', y=st.session_state.pv_name, color='Batch')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Data Before Alignment")
            fig = px.line(st.session_state.processed_data, x='Time', y=st.session_state.pv_name, color='Batch')
            st.plotly_chart(fig, use_container_width=True)
            st.info("Run 'Align Data' from the sidebar to see the results.")

# ==============================================================================
# 3. 분석 구간 설정 페이지
# ==============================================================================
elif page == "3. Analysis Range":
    # --- 상태 초기화 ---
    if 'slice_ranges' not in st.session_state:
        st.session_state.slice_ranges = {}
    if 'time_offsets' not in st.session_state:
        st.session_state.time_offsets = {}
    if 'selected_batch_for_edit' not in st.session_state:
        st.session_state.selected_batch_for_edit = None

    if st.session_state.aligned_data is None:
        st.warning("Please align data in Step 2 first.")
    else:
        aligned_df = st.session_state.aligned_data.copy()
        aligned_df['Batch'] = aligned_df['Batch'].astype(str)
        all_batches = sorted(aligned_df['Batch'].unique().tolist())
        pv_name = st.session_state.pv_name

        # --- 사이드바: 배치 선택 및 편집 ---
        st.sidebar.header("Step 3: Edit Batches")
        selected_batch = st.sidebar.selectbox(
            "Select a batch to edit:",
            ["None"] + all_batches,
            index=0
        )
        
        st.session_state.selected_batch_for_edit = selected_batch if selected_batch != "None" else None

        if st.session_state.selected_batch_for_edit:
            batch_id = st.session_state.selected_batch_for_edit
            batch_df = aligned_df[aligned_df['Batch'] == batch_id]
            min_time, max_time = float(batch_df['Aligned_Time'].min()), float(batch_df['Aligned_Time'].max())

            st.sidebar.subheader(f"Editing Batch: {batch_id}")
            
            # 1. 시간 축 이동 (슬라이더 + 숫자 입력)
            st.sidebar.markdown("<h6>Time Shift</h6>", unsafe_allow_html=True)
            duration = max_time - min_time
            shift_range = duration / 2.0
            
            current_offset = float(st.session_state.time_offsets.get(batch_id, 0.0))
            
            # 숫자 입력과 슬라이더를 연동
            offset_from_input = st.sidebar.number_input(
                "Shift Value",
                value=current_offset,
                step=1.0,
                key=f"offset_input_{batch_id}",
                label_visibility="collapsed"
            )
            
            offset_from_slider = st.sidebar.slider(
                "Shift Slider",
                min_value=float(-shift_range),
                max_value=float(shift_range),
                value=float(offset_from_input), # 숫자 입력값으로 슬라이더 위치 설정
                step=1.0,
                key=f"offset_slider_{batch_id}",
                label_visibility="collapsed"
            )
            st.session_state.time_offsets[batch_id] = offset_from_slider

            # 2. 분석 구간 (슬라이더 + 숫자 입력)
            st.sidebar.markdown("<h6>Analysis Range (Slicing)</h6>", unsafe_allow_html=True)
            current_start, current_end = st.session_state.slice_ranges.get(
                batch_id,
                (min_time, max_time)
            )

            # 숫자 입력과 슬라이더를 연동
            col_start, col_end = st.sidebar.columns(2)
            with col_start:
                start_from_input = st.number_input("Start", value=float(current_start), min_value=min_time, max_value=max_time, step=1.0)
            with col_end:
                end_from_input = st.number_input("End", value=float(current_end), min_value=min_time, max_value=max_time, step=1.0)

            range_from_slider = st.sidebar.slider(
                "Slicing Slider",
                min_value=min_time,
                max_value=max_time,
                value=(float(start_from_input), float(end_from_input)), # 숫자 입력값으로 슬라이더 위치 설정
                step=1.0,
                key=f"range_{batch_id}",
                label_visibility="collapsed"
            )
            st.session_state.slice_ranges[batch_id] = range_from_slider

        # --- 사이드바: 전역 편집 ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Global Edit (for All Batches)")

        # 전체 데이터의 시간 범위 계산 및 float으로 강제 변환
        global_min_time = float(aligned_df['Aligned_Time'].min())
        global_max_time = float(aligned_df['Aligned_Time'].max())
        global_duration = global_max_time - global_min_time

        # 전역 시간 축 이동
        st.sidebar.markdown("<h6>Global Time Shift</h6>", unsafe_allow_html=True)
        global_offset_input = st.sidebar.number_input("Global Shift Value", value=0.0, step=1.0, label_visibility="collapsed")
        global_offset_slider = st.sidebar.slider(
            "Global Shift Slider",
            min_value=float(-(global_duration / 2)),
            max_value=float(global_duration / 2),
            value=float(global_offset_input),
            step=1.0,
            key="global_offset_slider",
            label_visibility="collapsed"
        )
        
        # 전역 분석 구간
        st.sidebar.markdown("<h6>Global Analysis Range</h6>", unsafe_allow_html=True)
        g_col_start, g_col_end = st.sidebar.columns(2)
        with g_col_start:
            global_start_input = st.number_input("Global Start", value=global_min_time, min_value=global_min_time, max_value=global_max_time, step=1.0)
        with g_col_end:
            global_end_input = st.number_input("Global End", value=global_max_time, min_value=global_min_time, max_value=global_max_time, step=1.0)
            
        global_range_slider = st.sidebar.slider(
            "Global Slicing Slider",
            min_value=global_min_time,
            max_value=global_max_time,
            value=(float(global_start_input), float(global_end_input)),
            step=1.0,
            key="global_range_slider",
            label_visibility="collapsed"
        )

        if st.sidebar.button("Apply Global Changes to All Batches"):
            for batch_id in all_batches:
                st.session_state.time_offsets[batch_id] = global_offset_slider
                st.session_state.slice_ranges[batch_id] = global_range_slider
            st.success("Global changes have been applied to all batches.")
            st.rerun()

        # --- 메인 화면: 통합 오버레이 그래프 ---
        st.header("Overlayed Batch Profiles")
        st.info("All batches are shown with their current time shifts and slices applied. Use the sidebar to edit.")
        
        fig_overlay = go.Figure()

        for batch_id in all_batches:
            batch_df = aligned_df[aligned_df['Batch'] == batch_id].copy()
            
            # 1. 시간 이동 적용
            offset = st.session_state.time_offsets.get(batch_id, 0.0)
            batch_df['Shifted_Time'] = batch_df['Aligned_Time'] + offset

            # 2. 분석 구간 적용
            min_val, max_val = float(batch_df['Aligned_Time'].min()), float(batch_df['Aligned_Time'].max())
            start, end = st.session_state.slice_ranges.get(batch_id, (min_val, max_val))
            
            # 슬라이싱은 원본 Aligned_Time 기준
            sliced_df = batch_df[(batch_df['Aligned_Time'] >= start) & (batch_df['Aligned_Time'] <= end)]

            if not sliced_df.empty:
                fig_overlay.add_trace(go.Scatter(
                    x=sliced_df['Shifted_Time'],
                    y=sliced_df[pv_name],
                    mode='lines',
                    name=batch_id,
                    line=dict(width=4 if batch_id == st.session_state.selected_batch_for_edit else 2) # 선택된 배치 하이라이트
                ))
        
        fig_overlay.update_layout(height=600)
        st.plotly_chart(fig_overlay, use_container_width=True)

        # --- 최종 분석 데이터 생성 ---
        st.sidebar.markdown("---")
        if st.sidebar.button("Generate Final Analysis Data", type="primary"):
            with st.spinner("Generating final data with all shifts and slices..."):
                final_dfs = []
                for batch_id in all_batches:
                    batch_df = aligned_df[aligned_df['Batch'] == batch_id].copy()
                    offset = st.session_state.time_offsets.get(batch_id, 0.0)
                    batch_df['Final_Time'] = batch_df['Aligned_Time'] + offset
                    
                    min_val, max_val = float(batch_df['Aligned_Time'].min()), float(batch_df['Aligned_Time'].max())
                    start, end = st.session_state.slice_ranges.get(batch_id, (min_val, max_val))

                    sliced_df = batch_df[(batch_df['Aligned_Time'] >= start) & (batch_df['Aligned_Time'] <= end)]
                    
                    if not sliced_df.empty:
                        # 최종 데이터는 Final_Time 컬럼과 PV 컬럼만 유지
                        final_dfs.append(sliced_df[['Batch', 'Final_Time', pv_name]].rename(columns={'Final_Time': 'Aligned_Time'}))
                
                if final_dfs:
                    final_analysis_data = pd.concat(final_dfs, ignore_index=True)
                    st.session_state.analysis_data = final_analysis_data
                    st.success("Final analysis data generated successfully!")
                    st.rerun()
                else:
                    st.error("No data available after transformations.")

        if st.session_state.analysis_data is not None:
            st.sidebar.success("Analysis Data is ready for the next steps!")

# ==============================================================================
# 4. 골든 배치 페이지
# ==============================================================================
elif page == "4. Golden Batch":
    if st.session_state.analysis_data is None:
        st.warning("Please define and apply an analysis range in Step 3 first.")
    else:
        st.sidebar.header("Step 4: Golden Batch Analysis")
        quality_data = st.session_state.quality_data
        quality_metrics = [col for col in quality_data.columns if col != 'Batch']
        
        st.session_state.target_quality = st.sidebar.selectbox("Select target quality metric:", quality_metrics)
        st.session_state.optimization_direction = st.sidebar.radio("Optimization direction:", ('Maximize', 'Minimize'), horizontal=True)

        top_5_batches = quality_data.nlargest(5, st.session_state.target_quality)['Batch'].tolist() if st.session_state.optimization_direction == 'Maximize' else quality_data.nsmallest(5, st.session_state.target_quality)['Batch'].tolist()
        
        st.sidebar.write("Top 5 Recommended Batches:")
        st.sidebar.dataframe(quality_data[quality_data['Batch'].isin(top_5_batches)])
        
        selected_golden_batches = st.sidebar.multiselect("Select batches for Golden Profile:", quality_data['Batch'].unique().tolist(), default=top_5_batches)

        if st.sidebar.button("Analyze Golden Batch"):
            if selected_golden_batches:
                analysis_data = st.session_state.analysis_data
                pv_name = st.session_state.pv_name
                golden_batch_data = analysis_data[analysis_data['Batch'].isin(selected_golden_batches)]
                golden_profile = golden_batch_data.groupby('Aligned_Time')[pv_name].agg(['mean', 'std']).reset_index()
                golden_profile['upper_bound'] = golden_profile['mean'] + golden_profile['std']
                golden_profile['lower_bound'] = golden_profile['mean'] - golden_profile['std']
                st.session_state.golden_batch_profile = golden_profile
                st.success("Golden Batch analysis complete!")
        
        # --- 메인 화면 출력 ---
        st.header("4. Golden Batch Analysis Results")
        if st.session_state.golden_batch_profile is not None:
            profile_df = st.session_state.golden_batch_profile
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=profile_df['Aligned_Time'], y=profile_df['mean'], mode='lines', name='Golden Profile (Mean)', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=profile_df['Aligned_Time'], y=profile_df['upper_bound'], mode='lines', name='Upper Bound', line=dict(width=0)))
            fig.add_trace(go.Scatter(x=profile_df['Aligned_Time'], y=profile_df['lower_bound'], mode='lines', name='Lower Bound', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0)))
            fig.update_layout(title='Golden Profile with Operating Range', xaxis_title='Aligned Time', yaxis_title=st.session_state.pv_name)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select batches and run 'Analyze Golden Batch' from the sidebar.")

# ==============================================================================
# 5. PLS & 최적화 페이지
# ==============================================================================
elif page == "5. PLS & Optimization":
    if st.session_state.golden_batch_profile is None:
        st.warning("Please complete the Golden Batch analysis in Step 4 first.")
    else:
        st.sidebar.header("Step 5: PLS & Optimization")
        
        # PLS Analysis
        n_components = st.sidebar.slider("Number of PLS Components", 1, 10, 2)
        if st.sidebar.button("Run PLS Analysis"):
            analysis_data = st.session_state.analysis_data
            quality_data = st.session_state.quality_data
            pv_name = st.session_state.pv_name
            X_wide = analysis_data.pivot(index='Batch', columns='Aligned_Time', values=pv_name)
            X_wide.fillna(method='ffill', axis=1, inplace=True)
            X_wide.fillna(method='bfill', axis=1, inplace=True)
            X_wide.fillna(0, inplace=True)
            y = quality_data.set_index('Batch').loc[X_wide.index, st.session_state.target_quality]
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_wide, y)
            st.session_state.pls_coefficients = pd.DataFrame({'Aligned_Time': X_wide.columns, 'Coefficient': pls.coef_.flatten()})
            st.success("PLS analysis complete!")

        # Optimized Profile Suggestion
        if st.session_state.pls_coefficients is not None:
            st.sidebar.subheader("Optimized Profile")
            optimization_strength = st.sidebar.slider("Optimization Strength", 0.1, 2.0, 1.0, 0.1)
            if st.sidebar.button("Suggest Optimized Profile"):
                golden_profile = st.session_state.golden_batch_profile.copy()
                pls_coeffs = st.session_state.pls_coefficients.copy()
                merged_profile = pd.merge(golden_profile, pls_coeffs, on='Aligned_Time')
                if st.session_state.optimization_direction == 'Maximize':
                    merged_profile['optimized'] = merged_profile['mean'] + (merged_profile['Coefficient'] * optimization_strength)
                else:
                    merged_profile['optimized'] = merged_profile['mean'] - (merged_profile['Coefficient'] * optimization_strength)
                st.session_state.optimized_profile = merged_profile
                st.success("Optimized profile suggestion complete!")
        
        # --- 메인 화면 출력 ---
        st.header("5. PLS & Optimization Results")
        if st.session_state.pls_coefficients is not None:
            st.subheader("PLS Regression Coefficients")
            fig_pls = px.bar(st.session_state.pls_coefficients, x='Aligned_Time', y='Coefficient', title='Influence of Each Time Point on Quality')
            st.plotly_chart(fig_pls, use_container_width=True)
        else:
             st.info("Run 'PLS Analysis' from the sidebar to see the results.")

        if st.session_state.optimized_profile is not None:
            st.subheader("Golden Profile vs. Optimized Profile")
            profile_df = st.session_state.optimized_profile
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=profile_df['Aligned_Time'], y=profile_df['mean'], mode='lines', name='Golden Profile', line=dict(color='blue', dash='dash')))
            fig_opt.add_trace(go.Scatter(x=profile_df['Aligned_Time'], y=profile_df['optimized'], mode='lines', name='Optimized Profile', line=dict(color='green', width=3)))
            fig_opt.update_layout(title='Comparison: Golden Profile and Optimized Profile', xaxis_title='Aligned Time', yaxis_title=st.session_state.pv_name)
            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.info("Run 'Suggest Optimized Profile' from the sidebar to see the results.") 