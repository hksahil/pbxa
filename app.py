import streamlit as st
import pandas as pd
import io
from src.report import extract_report_metadata
from src.utils import filter_dataframe_by_text

try:
    from pbixray.core import PBIXRay
except ImportError:
    PBIXRay = None

# Page configuration
st.set_page_config(page_title="PBIX Analyser", page_icon="📊", layout="wide")

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def display_filterable_dataframe(df, key=None, hide_index=True, use_container_width=True):
    """
    Display a DataFrame with enhanced column filtering capabilities.
    
    Args:
        df: pandas DataFrame to display
        key: Unique key for the dataframe widget
        hide_index: Whether to hide the index column
        use_container_width: Whether to use full container width
    """
    if df is None or df.empty:
        return
    
    # Create an expander for column filters
    with st.expander("🔧 Column Filters", expanded=False):
        # Create multiselect filters for each column
        filtered_df = df.copy()
        
        # Create columns for filter layout (3 filters per row)
        num_cols = len(df.columns)
        cols_per_row = 3
        
        for i in range(0, num_cols, cols_per_row):
            filter_cols = st.columns(cols_per_row)
            
            for j, col_name in enumerate(df.columns[i:i+cols_per_row]):
                with filter_cols[j]:
                    # Get unique values for the column (limit to prevent performance issues)
                    unique_values = df[col_name].astype(str).unique()
                    
                    # Limit the number of unique values shown in filter
                    if len(unique_values) > 100:
                        st.caption(f"**{col_name}** (too many values to filter)")
                    else:
                        # Sort unique values
                        unique_values_sorted = sorted(unique_values)
                        
                        # Create multiselect filter
                        selected_values = st.multiselect(
                            f"Filter {col_name}",
                            options=unique_values_sorted,
                            default=None,
                            key=f"{key}_{col_name}_filter" if key else None,
                            label_visibility="collapsed",
                            placeholder=f"Filter {col_name}..."
                        )
                        
                        # Apply filter if values are selected
                        if selected_values:
                            filtered_df = filtered_df[filtered_df[col_name].astype(str).isin(selected_values)]
        
        # Show filter results summary
        if len(filtered_df) < len(df):
            st.info(f"📊 Showing {len(filtered_df)} of {len(df)} rows after column filtering")
    
    # Display the filtered dataframe
    st.dataframe(
        filtered_df,
        hide_index=hide_index,
        use_container_width=use_container_width,
        key=f"{key}_display" if key else None
    )

def compute_suspect_flag_export(df: pd.DataFrame) -> pd.DataFrame:
    keywords = [
        "eh",
        "essential home",
        "essential homes",
        "bu",
        "functional"
    ]
    def row_flag(row: pd.Series) -> str:
        text_parts = [
            str(row.get('Field Display Name', '')),
            str(row.get('Field Query Name', '')),
            str(row.get('Visual Title', '')),
        ]
        haystack = " ".join(text_parts).lower()
        return "Yes" if any(kw in haystack for kw in keywords) else "No"
    df = df.copy()
    if not df.empty:
        df['Suspect for change?'] = df.apply(row_flag, axis=1)
    return df


def append_report_name_to_df(df, report_name):
    """
    Add Report Name column as the first column in a DataFrame.
    
    Args:
        df: pandas DataFrame to modify
        report_name: Name of the report to add
        
    Returns:
        DataFrame with Report Name column added
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    df.insert(0, 'Report Name', report_name)
    return df


def build_report_summaries(uploaded_file, model=None) -> dict:
    if extract_report_metadata is None:
        return {}
    # Rewind file pointer if possible
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    report_data = extract_report_metadata(uploaded_file)
    if not report_data:
        return {}

    summary = report_data.get("summary", {})
    pages = report_data.get("pages", [])
    visuals = report_data.get("visuals", [])

    df_visuals = pd.DataFrame(visuals)
    if not df_visuals.empty and 'Visual ID' in df_visuals.columns:
        total_unique_visuals = df_visuals['Visual ID'].nunique()
        
        # Count slicers - separate direct and indirect
        direct_slicers = 0
        indirect_slicers = 0
        slicer_visuals = df_visuals[df_visuals['Visual Type'].str.lower().str.contains('slicer', na=False)]
        if not slicer_visuals.empty:
            # Direct slicers (not hidden)
            if 'Hidden' in slicer_visuals.columns:
                direct_slicers = slicer_visuals[slicer_visuals['Hidden'] == 'No']['Visual ID'].nunique()
                # Indirect slicers (hidden)
                indirect_slicers = slicer_visuals[slicer_visuals['Hidden'] == 'Yes']['Visual ID'].nunique()
            else:
                # If Hidden column doesn't exist, treat all slicers as direct
                direct_slicers = slicer_visuals['Visual ID'].nunique()
        
        total_slicers = direct_slicers + indirect_slicers
        non_slicer_visuals = total_unique_visuals - total_slicers
    else:
        non_slicer_visuals = 0
        direct_slicers = 0
        indirect_slicers = 0

    # Get PBIXRay metrics if model is available
    tables_count = 0
    dax_measures_count = 0
    columns_count = 0
    has_incremental_refresh = "No"
    has_row_level_security = "No"
    
    if model is not None:
        try:
            # Tables count from Power Query - count unique table names
            power_query_df = getattr(model, 'power_query', pd.DataFrame())
            if not power_query_df.empty:
                # Try different possible column names for table names
                table_col = None
                for col_name in ['TableName', 'Table', 'Name']:
                    if col_name in power_query_df.columns:
                        table_col = col_name
                        break
                
                if table_col:
                    tables_count = power_query_df[table_col].nunique()
                else:
                    tables_count = 0
            else:
                tables_count = 0
        except Exception:
            tables_count = 0
            
        try:
            # DAX Measures count from DAX Measures Analysis
            dax_measures_df = getattr(model, 'dax_measures', pd.DataFrame())
            if not dax_measures_df.empty and 'Name' in dax_measures_df.columns:
                dax_measures_count = len(dax_measures_df['Name'].dropna())
            else:
                dax_measures_count = 0
        except Exception:
            dax_measures_count = 0
            
        try:
            # Columns count from Column Analysis (statistics table)
            statistics_df = getattr(model, 'statistics', pd.DataFrame())
            columns_count = len(statistics_df) if not statistics_df.empty else 0
        except Exception:
            columns_count = 0
            
        try:
            # Check for Incremental Refresh (RangeStart parameter)
            m_parameters_df = getattr(model, 'm_parameters', pd.DataFrame())
            if not m_parameters_df.empty:
                # Try different possible column names for parameter names
                name_col = None
                for col_name in ['ParameterName', 'Name', 'Parameter']:
                    if col_name in m_parameters_df.columns:
                        name_col = col_name
                        break
                
                if name_col:
                    # Check if any parameter name contains 'RangeStart'
                    if m_parameters_df[name_col].astype(str).str.contains('RangeStart', case=False, na=False).any():
                        has_incremental_refresh = "Yes"
        except Exception:
            pass
            
        try:
            # Check for Row-Level Security (RLS)
            rls_df = getattr(model, 'rls', pd.DataFrame())
            if not rls_df.empty:
                has_row_level_security = "Yes"
        except Exception:
            pass

    # Filters count
    total_filters = 0
    for p in pages:
        pf = p.get('Page Filters', '')
        if pf and pf != 'None':
            total_filters += len(pf.split(' | '))
    uniq_vis_filters = df_visuals[['Visual ID', 'Visual Filters']].drop_duplicates() if not df_visuals.empty else pd.DataFrame()
    for _, v in uniq_vis_filters.iterrows():
        vf = v.get('Visual Filters', '')
        if vf:
            total_filters += len(vf.split(' | '))

    # Bookmarks count (definitions and references)
    total_bookmarks = sum(p.get('Bookmarks', 0) for p in pages)
    total_bookmark_references = sum(p.get('Bookmark References', 0) for p in pages)

    # Report summary dataframe with requested columns
    report_name = getattr(uploaded_file, 'name', 'Report').replace('.pbix', '')
    df_report_summary = pd.DataFrame([
        {
            'Report Name': report_name,
            'Pages': summary.get('Total Pages', 0),
            'Visuals': non_slicer_visuals,
            'Direct Slicers': direct_slicers,
            'Indirect Slicers': indirect_slicers,
            'Static Elements': summary.get('Total Static Elements', 0),
            'Filters': total_filters,
            'Tables': tables_count,
            'DAX Measures': dax_measures_count,
            'Columns': columns_count,
            'Bookmarks': total_bookmarks,
            'Bookmark References': total_bookmark_references,
            'Incremental Refresh?': has_incremental_refresh,
            'Row Level Security': has_row_level_security,
        }
    ])

    # Page summary dataframe (ordered)
    df_pages = pd.DataFrame(pages)
    if not df_pages.empty:
        if not df_visuals.empty and 'Visual ID' in df_visuals.columns:
            visual_ids_by_page = df_visuals.groupby('Page Name')['Visual ID'].nunique()
            
            # Separate direct and indirect slicers by page
            slicer_visuals = df_visuals[df_visuals['Visual Type'].str.lower().str.contains('slicer', na=False)]
            if not slicer_visuals.empty and 'Hidden' in slicer_visuals.columns:
                direct_slicers_by_page = slicer_visuals[slicer_visuals['Hidden'] == 'No'].groupby('Page Name')['Visual ID'].nunique()
                indirect_slicers_by_page = slicer_visuals[slicer_visuals['Hidden'] == 'Yes'].groupby('Page Name')['Visual ID'].nunique()
            else:
                # If Hidden column doesn't exist, treat all slicers as direct
                direct_slicers_by_page = slicer_visuals.groupby('Page Name')['Visual ID'].nunique() if not slicer_visuals.empty else pd.Series(dtype=int)
                indirect_slicers_by_page = pd.Series(dtype=int)
            
            total_slicers_by_page = direct_slicers_by_page.add(indirect_slicers_by_page, fill_value=0)
            non_slicer_by_page = (visual_ids_by_page - total_slicers_by_page).fillna(0).astype(int)
        else:
            direct_slicers_by_page = pd.Series(dtype=int)
            indirect_slicers_by_page = pd.Series(dtype=int)
            non_slicer_by_page = pd.Series(dtype=int)

        df_pages = df_pages.copy()
        df_pages['Direct Slicers'] = df_pages['Page Name'].map(direct_slicers_by_page).fillna(0).astype(int)
        df_pages['Indirect Slicers'] = df_pages['Page Name'].map(indirect_slicers_by_page).fillna(0).astype(int)
        df_pages['Visual Count(no slicers)'] = df_pages['Page Name'].map(non_slicer_by_page).fillna(0).astype(int)
        desired_order_pages = ['Page Name', 'All Elements', 'Direct Slicers', 'Indirect Slicers', 'Static Elements', 'Visual Count(no slicers)', 'Bookmarks', 'Bookmark References', 'Page Filters', 'Groups']
        df_pages_summary = df_pages.reindex(columns=[c for c in desired_order_pages if c in df_pages.columns])
    else:
        df_pages_summary = pd.DataFrame(columns=['Page Name', 'All Elements', 'Direct Slicers', 'Indirect Slicers', 'Static Elements', 'Visual Count(no slicers)', 'Bookmarks', 'Bookmark References', 'Page Filters', 'Groups'])

    # Visual summary dataframe with Suspect flag
    df_visuals_full = pd.DataFrame(visuals)
    if not df_visuals_full.empty:
        df_visuals_summary = compute_suspect_flag_export(df_visuals_full)
        desired_visual_cols = [
            'Page Name', 'Visual Title', 'Visual Type', 'Field Display Name', 'Field Query Name',
            'Field Type', 'Field Format', 'Is Measure', 'Aggregation', 'Projection Type',
            'Active', 'Suspect for change?'
        ]
        df_visuals_summary = df_visuals_summary.reindex(columns=[c for c in desired_visual_cols if c in df_visuals_summary.columns])
        df_visuals_summary.rename(columns={
            'Page Name': 'Page',
            'Visual Title': 'Title',
            'Field Display Name': 'Field',
            'Is Measure': 'Measure?',
            'Aggregation': 'Aggregation',
            'Projection Type': 'Projection Type',
        }, inplace=True)
    else:
        df_visuals_summary = pd.DataFrame(columns=[
            'Page', 'Title', 'Visual Type', 'Field', 'Field Query Name', 'Type',
            'Field Format', 'Measure?', 'Aggregation', 'Projection Type', 'Active', 'Suspect for change?'
        ])

    return {
        'report_summary': df_report_summary,
        'page_summary': df_pages_summary,
        'visual_summary': df_visuals_summary,
    }


def process_multiple_files(uploaded_files):
    """
    Process multiple PBIX files and aggregate all results.
    
    Args:
        uploaded_files: List of uploaded PBIX files
    """
    st.info(f"Processing {len(uploaded_files)} PBIX files...")
    
    # Initialize aggregated dataframes
    all_report_summaries = []
    all_page_summaries = []
    all_visual_summaries = []
    all_table_analysis = []
    all_relationships = []
    all_columns = []
    all_m_parameters = []
    all_dax_tables = []
    all_calculated_columns = []
    all_dax_measures = []
    all_rls = []
    
    # Process each file with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        report_name = uploaded_file.name.replace('.pbix', '')
        status_text.text(f"Processing: {report_name} ({idx + 1}/{len(uploaded_files)})")
        
        try:
            # Rewind file pointer
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            
            # Initialize model
            model = None
            if PBIXRay is not None:
                try:
                    model = PBIXRay(uploaded_file)
                except Exception as e:
                    st.warning(f"PBIXRay failed for {report_name}: {str(e)}")
            
            # Get report summaries
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            
            summaries = build_report_summaries(uploaded_file, model)
            
            # Append report summary
            if summaries and 'report_summary' in summaries and not summaries['report_summary'].empty:
                all_report_summaries.append(summaries['report_summary'])
            
            # Append page summary
            if summaries and 'page_summary' in summaries and not summaries['page_summary'].empty:
                df_page = append_report_name_to_df(summaries['page_summary'], report_name)
                all_page_summaries.append(df_page)
            
            # Append visual summary
            if summaries and 'visual_summary' in summaries and not summaries['visual_summary'].empty:
                df_visual = append_report_name_to_df(summaries['visual_summary'], report_name)
                all_visual_summaries.append(df_visual)
            
            # Append model data
            if model is not None:
                try:
                    power_query = getattr(model, 'power_query', pd.DataFrame())
                    if not power_query.empty:
                        df_pq = append_report_name_to_df(power_query, report_name)
                        all_table_analysis.append(df_pq)
                except Exception:
                    pass
                
                try:
                    relationships = getattr(model, 'relationships', pd.DataFrame())
                    if not relationships.empty:
                        df_rel = append_report_name_to_df(relationships, report_name)
                        all_relationships.append(df_rel)
                except Exception:
                    pass
                
                try:
                    statistics_df = getattr(model, 'statistics', pd.DataFrame())
                    if not statistics_df.empty:
                        df_stats = append_report_name_to_df(statistics_df, report_name)
                        all_columns.append(df_stats)
                except Exception:
                    pass
                
                try:
                    m_parameters = getattr(model, 'm_parameters', pd.DataFrame())
                    if not m_parameters.empty:
                        df_m = append_report_name_to_df(m_parameters, report_name)
                        all_m_parameters.append(df_m)
                except Exception:
                    pass
                
                try:
                    dax_tables = getattr(model, 'dax_tables', pd.DataFrame())
                    if not dax_tables.empty:
                        df_dax_t = append_report_name_to_df(dax_tables, report_name)
                        all_dax_tables.append(df_dax_t)
                except Exception:
                    pass
                
                try:
                    calculated_cols = None
                    for attr_name in ['calculated_columns', 'dax_columns', 'columns', 'dax_columns_df']:
                        if hasattr(model, attr_name):
                            attr_data = getattr(model, attr_name)
                            if hasattr(attr_data, 'size') and attr_data.size > 0:
                                calculated_cols = attr_data
                                break
                    
                    if calculated_cols is not None and not calculated_cols.empty:
                        df_calc = append_report_name_to_df(calculated_cols, report_name)
                        all_calculated_columns.append(df_calc)
                except Exception:
                    pass
                
                try:
                    dax_measures = getattr(model, 'dax_measures', pd.DataFrame())
                    if not dax_measures.empty:
                        df_dax_m = append_report_name_to_df(dax_measures, report_name)
                        all_dax_measures.append(df_dax_m)
                except Exception:
                    pass
                
                try:
                    rls_df = getattr(model, 'rls', pd.DataFrame())
                    if not rls_df.empty:
                        df_rls = append_report_name_to_df(rls_df, report_name)
                        all_rls.append(df_rls)
                except Exception:
                    pass
        
        except Exception as e:
            st.error(f"Error processing {report_name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("✅ All files processed!")
    st.success(f"Successfully processed {len(uploaded_files)} PBIX files")
    
    # Combine all dataframes and store in session state
    st.session_state.multi_combined_report_summary = pd.concat(all_report_summaries, ignore_index=True) if all_report_summaries else pd.DataFrame()
    st.session_state.multi_combined_page_summary = pd.concat(all_page_summaries, ignore_index=True) if all_page_summaries else pd.DataFrame()
    st.session_state.multi_combined_visual_summary = pd.concat(all_visual_summaries, ignore_index=True) if all_visual_summaries else pd.DataFrame()
    st.session_state.multi_combined_table_analysis = pd.concat(all_table_analysis, ignore_index=True) if all_table_analysis else pd.DataFrame()
    st.session_state.multi_combined_relationships = pd.concat(all_relationships, ignore_index=True) if all_relationships else pd.DataFrame()
    st.session_state.multi_combined_columns = pd.concat(all_columns, ignore_index=True) if all_columns else pd.DataFrame()
    st.session_state.multi_combined_m_parameters = pd.concat(all_m_parameters, ignore_index=True) if all_m_parameters else pd.DataFrame()
    st.session_state.multi_combined_dax_tables = pd.concat(all_dax_tables, ignore_index=True) if all_dax_tables else pd.DataFrame()
    st.session_state.multi_combined_calculated_columns = pd.concat(all_calculated_columns, ignore_index=True) if all_calculated_columns else pd.DataFrame()
    st.session_state.multi_combined_dax_measures = pd.concat(all_dax_measures, ignore_index=True) if all_dax_measures else pd.DataFrame()
    st.session_state.multi_combined_rls = pd.concat(all_rls, ignore_index=True) if all_rls else pd.DataFrame()
    st.session_state.multi_num_files = len(uploaded_files)
    


def display_multi_file_results():
    """
    Display the aggregated results from multi-file processing (stored in session state).
    """
    # Get data from session state
    combined_report_summary = st.session_state.get('multi_combined_report_summary', pd.DataFrame())
    combined_page_summary = st.session_state.get('multi_combined_page_summary', pd.DataFrame())
    combined_visual_summary = st.session_state.get('multi_combined_visual_summary', pd.DataFrame())
    combined_table_analysis = st.session_state.get('multi_combined_table_analysis', pd.DataFrame())
    combined_relationships = st.session_state.get('multi_combined_relationships', pd.DataFrame())
    combined_columns = st.session_state.get('multi_combined_columns', pd.DataFrame())
    combined_m_parameters = st.session_state.get('multi_combined_m_parameters', pd.DataFrame())
    combined_dax_tables = st.session_state.get('multi_combined_dax_tables', pd.DataFrame())
    combined_calculated_columns = st.session_state.get('multi_combined_calculated_columns', pd.DataFrame())
    combined_dax_measures = st.session_state.get('multi_combined_dax_measures', pd.DataFrame())
    combined_rls = st.session_state.get('multi_combined_rls', pd.DataFrame())
    num_files = st.session_state.get('multi_num_files', 0)
    
    # Display aggregated results
    st.divider()
    st.header("📊 Aggregated Analysis Results")
    
    # 1. Report Summary (first) - NO FILTERING
    st.subheader("Report Summary")
    if not combined_report_summary.empty:
        display_filterable_dataframe(combined_report_summary, key="multi_report_summary", hide_index=True)
    else:
        st.info("No report summary data available.")
    
    # Add global text filter
    st.divider()
    st.markdown("### 🔍 Filter All Tables Below")
    search_text = st.text_input(
        "Search across all aggregated tables:",
        placeholder="Enter text to filter all rows in tables below...",
        help="This will filter all tables below based on your search text (case-insensitive)",
        key="multi_search_text"
    )
    
    if search_text:
        st.info(f"🔎 Filtering all tables below for: '{search_text}'")
    
    st.divider()
    
    # 2. Table Analysis
    if not combined_table_analysis.empty:
        st.subheader("Table Analysis (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_table_analysis, search_text)
        display_filterable_dataframe(filtered_df, key="multi_table_analysis")
        if search_text and len(filtered_df) < len(combined_table_analysis):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_table_analysis)} rows")
    
    # 3. Relationship Analysis
    if not combined_relationships.empty:
        st.subheader("Relationships Analysis (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_relationships, search_text)
        display_filterable_dataframe(filtered_df, key="multi_relationships")
        if search_text and len(filtered_df) < len(combined_relationships):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_relationships)} rows")
    
    # 4. Column Analysis
    if not combined_columns.empty:
        st.subheader("Column Analysis (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_columns, search_text)
        display_filterable_dataframe(filtered_df, key="multi_columns")
        if search_text and len(filtered_df) < len(combined_columns):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_columns)} rows")
    
    # 5. M Parameters
    if not combined_m_parameters.empty:
        st.subheader("M Parameters (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_m_parameters, search_text)
        display_filterable_dataframe(filtered_df, key="multi_m_parameters")
        if search_text and len(filtered_df) < len(combined_m_parameters):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_m_parameters)} rows")
    
    # 6. DAX Tables
    if not combined_dax_tables.empty:
        st.subheader("DAX Tables (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_dax_tables, search_text)
        display_filterable_dataframe(filtered_df, key="multi_dax_tables")
        if search_text and len(filtered_df) < len(combined_dax_tables):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_dax_tables)} rows")
    
    # 7. Calculated Columns
    if not combined_calculated_columns.empty:
        st.subheader("Calculated Columns (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_calculated_columns, search_text)
        display_filterable_dataframe(filtered_df, key="multi_calculated_columns")
        if search_text and len(filtered_df) < len(combined_calculated_columns):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_calculated_columns)} rows")
    
    # 8. DAX Measures
    if not combined_dax_measures.empty:
        st.subheader("DAX Measures (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_dax_measures, search_text)
        display_filterable_dataframe(filtered_df, key="multi_dax_measures")
        if search_text and len(filtered_df) < len(combined_dax_measures):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_dax_measures)} rows")
    
    # 9. Page Summary
    if not combined_page_summary.empty:
        st.subheader("Page Summary (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_page_summary, search_text)
        display_filterable_dataframe(filtered_df, key="multi_page_summary", hide_index=True)
        if search_text and len(filtered_df) < len(combined_page_summary):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_page_summary)} rows")
    
    # 10. Visual Summary
    if not combined_visual_summary.empty:
        st.subheader("Visual Summary (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_visual_summary, search_text)
        display_filterable_dataframe(filtered_df, key="multi_visual_summary", hide_index=True)
        if search_text and len(filtered_df) < len(combined_visual_summary):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_visual_summary)} rows")
    
    # 11. Row-Level Security
    if not combined_rls.empty:
        st.subheader("Row-Level Security (All Reports)")
        filtered_df = filter_dataframe_by_text(combined_rls, search_text)
        display_filterable_dataframe(filtered_df, key="multi_rls", hide_index=True)
        if search_text and len(filtered_df) < len(combined_rls):
            st.caption(f"Showing {len(filtered_df)} of {len(combined_rls)} rows")
    
    # Export All Aggregated Data
    st.divider()
    export_buffer = io.BytesIO()
    
    with pd.ExcelWriter(export_buffer, engine='openpyxl') as writer:
        if not combined_report_summary.empty:
            combined_report_summary.to_excel(writer, index=False, sheet_name='Report Summary')
        if not combined_table_analysis.empty:
            combined_table_analysis.to_excel(writer, index=False, sheet_name='Table Analysis')
        if not combined_relationships.empty:
            combined_relationships.to_excel(writer, index=False, sheet_name='Relationships Analysis')
        if not combined_columns.empty:
            combined_columns.to_excel(writer, index=False, sheet_name='Columns Analysis')
        if not combined_m_parameters.empty:
            combined_m_parameters.to_excel(writer, index=False, sheet_name='M Parameters')
        if not combined_dax_tables.empty:
            combined_dax_tables.to_excel(writer, index=False, sheet_name='DAX Tables')
        if not combined_calculated_columns.empty:
            combined_calculated_columns.to_excel(writer, index=False, sheet_name='Calculated Columns')
        if not combined_dax_measures.empty:
            combined_dax_measures.to_excel(writer, index=False, sheet_name='DAX Measures')
        if not combined_page_summary.empty:
            combined_page_summary.to_excel(writer, index=False, sheet_name='Page Summary')
        if not combined_visual_summary.empty:
            combined_visual_summary.to_excel(writer, index=False, sheet_name='Visual Summary')
        if not combined_rls.empty:
            combined_rls.to_excel(writer, index=False, sheet_name='Row-Level Security')
    
    st.download_button(
        label="📥 Download Aggregated Analysis (Excel)",
        data=export_buffer.getvalue(),
        file_name=f"multi_pbix_analysis_{num_files}_files.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def process_single_file(uploaded_file):
    model = None
    if PBIXRay is not None:
        model = PBIXRay(uploaded_file)
    else:
        st.warning("pbixray is not installed; showing extractor-based summaries only.")

    # 1. Report Summary (first) - NO FILTERING
    st.subheader("Report Summary")
    summaries = build_report_summaries(uploaded_file, model)
    if summaries:
        display_filterable_dataframe(summaries['report_summary'], key="single_report_summary", hide_index=True)
    else:
        st.info("Could not extract report summary from the PBIX file.")

    # Add global text filter below Report Summary
    st.divider()
    st.markdown("### 🔍 Filter All Tables Below")
    search_text = st.text_input(
        "Search across all tables/dataframes:",
        placeholder="Enter text to filter all rows in tables below...",
        help="This will filter all tables below based on your search text (case-insensitive)"
    )
    
    if search_text:
        st.info(f"🔎 Filtering all tables below for: '{search_text}'")
    
    st.divider()

    # 2. Table Analysis
    if model is not None:
        try:
            if getattr(model, 'power_query', pd.DataFrame()).size:
                st.subheader("Table Analysis")
                filtered_df = filter_dataframe_by_text(model.power_query, search_text)
                display_filterable_dataframe(filtered_df, key="single_table_analysis")
                if search_text and len(filtered_df) < len(model.power_query):
                    st.caption(f"Showing {len(filtered_df)} of {len(model.power_query)} rows")
            else:
                st.info("No Table Analysis found.")
        except Exception:
            st.info("Table Analysis not available.")

    # 3. Relationship Analysis
    if model is not None:
        try:
            if getattr(model, 'relationships', pd.DataFrame()).size:
                st.subheader("Relationships Analysis")
                filtered_df = filter_dataframe_by_text(model.relationships, search_text)
                display_filterable_dataframe(filtered_df, key="single_relationships")
                if search_text and len(filtered_df) < len(model.relationships):
                    st.caption(f"Showing {len(filtered_df)} of {len(model.relationships)} rows")
            else:
                st.info("No Relationships found.")
        except Exception:
            st.info("Relationships Analysis not available.")

    # 4. Column Analysis
    if model is not None:
        try:
            st.subheader("Column Analysis")
            statistics_df = getattr(model, 'statistics', pd.DataFrame())
            filtered_df = filter_dataframe_by_text(statistics_df, search_text)
            display_filterable_dataframe(filtered_df, key="single_columns")
            if search_text and len(filtered_df) < len(statistics_df):
                st.caption(f"Showing {len(filtered_df)} of {len(statistics_df)} rows")
        except Exception:
            st.info("Column Analysis not available.")

    # 5. M Parameters
    if model is not None:
        try:
            if getattr(model, 'm_parameters', pd.DataFrame()).size:
                st.subheader("M Parameters")
                filtered_df = filter_dataframe_by_text(model.m_parameters, search_text)
                display_filterable_dataframe(filtered_df, key="single_m_parameters")
                if search_text and len(filtered_df) < len(model.m_parameters):
                    st.caption(f"Showing {len(filtered_df)} of {len(model.m_parameters)} rows")
            else:
                st.info("No M Parameters found.")
        except Exception:
            st.info("M Parameters not available.")

    # 6. DAX Tables
    if model is not None:
        try:
            if getattr(model, 'dax_tables', pd.DataFrame()).size:
                st.subheader("DAX Tables")
                filtered_df = filter_dataframe_by_text(model.dax_tables, search_text)
                display_filterable_dataframe(filtered_df, key="single_dax_tables")
                if search_text and len(filtered_df) < len(model.dax_tables):
                    st.caption(f"Showing {len(filtered_df)} of {len(model.dax_tables)} rows")
            else:
                st.info("No DAX Tables found.")
        except Exception:
            st.info("DAX Tables not available.")

    # 7. Calculated Columns
    if model is not None:
        try:
            # Try different possible attribute names for calculated columns
            calculated_cols = None
            for attr_name in ['calculated_columns', 'dax_columns', 'columns', 'dax_columns_df']:
                if hasattr(model, attr_name):
                    attr_data = getattr(model, attr_name)
                    if hasattr(attr_data, 'size') and attr_data.size > 0:
                        calculated_cols = attr_data
                        break
            
            if calculated_cols is not None and calculated_cols.size > 0:
                st.subheader("Calculated Columns")
                filtered_df = filter_dataframe_by_text(calculated_cols, search_text)
                display_filterable_dataframe(filtered_df, key="single_calculated_columns")
                if search_text and len(filtered_df) < len(calculated_cols):
                    st.caption(f"Showing {len(filtered_df)} of {len(calculated_cols)} rows")
            else:
                st.info("No Calculated Columns found.")
        except Exception as e:
            st.info(f"Calculated Columns not available: {str(e)}")

    # 8. DAX Measures
    if model is not None:
        try:
            if getattr(model, 'dax_measures', pd.DataFrame()).size:
                st.subheader("DAX Measures")
                filtered_df = filter_dataframe_by_text(model.dax_measures, search_text)
                display_filterable_dataframe(filtered_df, key="single_dax_measures")
                if search_text and len(filtered_df) < len(model.dax_measures):
                    st.caption(f"Showing {len(filtered_df)} of {len(model.dax_measures)} rows")
            else:
                st.info("No DAX measures found.")
        except Exception:
            st.info("DAX Measures not available.")

    # 9. Page Summary
    if summaries:
        st.subheader("Page Summary")
        filtered_df = filter_dataframe_by_text(summaries['page_summary'], search_text)
        display_filterable_dataframe(filtered_df, key="single_page_summary", hide_index=True)
        if search_text and len(filtered_df) < len(summaries['page_summary']):
            st.caption(f"Showing {len(filtered_df)} of {len(summaries['page_summary'])} rows")

    # 10. Visual Summary
    if summaries:
        st.subheader("Visual Summary")
        filtered_df = filter_dataframe_by_text(summaries['visual_summary'], search_text)
        display_filterable_dataframe(filtered_df, key="single_visual_summary", hide_index=True)
        if search_text and len(filtered_df) < len(summaries['visual_summary']):
            st.caption(f"Showing {len(filtered_df)} of {len(summaries['visual_summary'])} rows")

    # 11. Row-Level Security (RLS)
    if model is not None:
        try:
            rls_df = getattr(model, 'rls', pd.DataFrame())
            if not rls_df.empty:
                st.subheader("Row-Level Security (RLS)")
                filtered_df = filter_dataframe_by_text(rls_df, search_text)
                display_filterable_dataframe(filtered_df, key="single_rls", hide_index=True)
                if search_text and len(filtered_df) < len(rls_df):
                    st.caption(f"Showing {len(filtered_df)} of {len(rls_df)} rows")
            else:
                st.info("No Row-Level Security found.")
        except Exception:
            st.info("Row-Level Security not available.")

    # Export All Sheets Button
    st.divider()
    export_buffer = io.BytesIO()
    
    with pd.ExcelWriter(export_buffer, engine='openpyxl') as writer:
        # Sheet 1: Report Summary
        if summaries and not summaries['report_summary'].empty:
            summaries['report_summary'].to_excel(writer, index=False, sheet_name='Report Summary')
        
        # Sheet 2: Table Analysis
        if model is not None:
            try:
                power_query = getattr(model, 'power_query', pd.DataFrame())
                if not power_query.empty:
                    power_query.to_excel(writer, index=False, sheet_name='Table Analysis')
            except Exception:
                pass
        
        # Sheet 3: Relationships Analysis
        if model is not None:
            try:
                relationships = getattr(model, 'relationships', pd.DataFrame())
                if not relationships.empty:
                    relationships.to_excel(writer, index=False, sheet_name='Relationships Analysis')
            except Exception:
                pass
        
        # Sheet 4: Columns Analysis
        if model is not None:
            try:
                statistics_df = getattr(model, 'statistics', pd.DataFrame())
                if not statistics_df.empty:
                    statistics_df.to_excel(writer, index=False, sheet_name='Columns Analysis')
            except Exception:
                pass
        
        # Sheet 5: M Parameters
        if model is not None:
            try:
                m_parameters = getattr(model, 'm_parameters', pd.DataFrame())
                if not m_parameters.empty:
                    m_parameters.to_excel(writer, index=False, sheet_name='M Parameters')
            except Exception:
                pass
        
        # Sheet 6: DAX Tables
        if model is not None:
            try:
                dax_tables = getattr(model, 'dax_tables', pd.DataFrame())
                if not dax_tables.empty:
                    dax_tables.to_excel(writer, index=False, sheet_name='DAX Tables')
            except Exception:
                pass
        
        # Sheet 7: Calculated Columns
        if model is not None:
            try:
                # Try different possible attribute names for calculated columns
                calculated_columns = None
                for attr_name in ['calculated_columns', 'dax_columns', 'columns', 'dax_columns_df']:
                    if hasattr(model, attr_name):
                        attr_data = getattr(model, attr_name)
                        if hasattr(attr_data, 'size') and attr_data.size > 0:
                            calculated_columns = attr_data
                            break
                
                if calculated_columns is not None and not calculated_columns.empty:
                    calculated_columns.to_excel(writer, index=False, sheet_name='Calculated Columns')
            except Exception:
                pass
        
        # Sheet 8: DAX Measures
        if model is not None:
            try:
                dax_measures = getattr(model, 'dax_measures', pd.DataFrame())
                if not dax_measures.empty:
                    dax_measures.to_excel(writer, index=False, sheet_name='DAX Measures')
            except Exception:
                pass
        
        # Sheet 9: Page Summary
        if summaries and not summaries['page_summary'].empty:
            summaries['page_summary'].to_excel(writer, index=False, sheet_name='Page Summary')
        
        # Sheet 10: Visual Summary
        if summaries and not summaries['visual_summary'].empty:
            summaries['visual_summary'].to_excel(writer, index=False, sheet_name='Visual Summary')
        
        # Sheet 11: Row-Level Security (RLS)
        if model is not None:
            try:
                rls_df = getattr(model, 'rls', pd.DataFrame())
                if not rls_df.empty:
                    rls_df.to_excel(writer, index=False, sheet_name='Row-Level Security')
            except Exception:
                pass
    
    st.download_button(
        label="📥 Download Complete Analysis (Excel)",
        data=export_buffer.getvalue(),
        file_name=f"{getattr(uploaded_file, 'name', 'analysis').replace('.pbix', '')}_complete_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Main app
st.title("📊 PBIX Analyser for Reckitt")

# Add analysis mode selector
analysis_mode = st.radio(
    "Select Analysis Mode:",
    options=["Single File Analysis", "Multi-File Analysis"],
    horizontal=True,
    help="Choose whether to analyze one PBIX file or multiple PBIX files at once"
)

st.divider()

if analysis_mode == "Single File Analysis":
    uploaded_file = st.file_uploader("📁 Upload a PBIX file", type="pbix")
    
    if uploaded_file:
        process_single_file(uploaded_file)
    else:
        st.info("Upload a PBIX file to get started.")
        
else:  # Multi-File Analysis
    uploaded_files = st.file_uploader(
        "📁 Upload multiple PBIX files",
        type="pbix",
        accept_multiple_files=True,
        help="Select multiple PBIX files to analyze them together"
    )
    
    # Check if new files were uploaded (different from processed files)
    uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    processed_file_names = st.session_state.get('multi_processed_file_names', [])
    files_changed = uploaded_file_names != processed_file_names
    
    # Clear session state if files changed
    if files_changed and 'multi_combined_report_summary' in st.session_state:
        # Clear all multi-file session state
        for key in list(st.session_state.keys()):
            if key.startswith('multi_'):
                del st.session_state[key]
    
    if uploaded_files and len(uploaded_files) > 0:
        st.info(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Show selected files
        with st.expander("View selected files"):
            for i, f in enumerate(uploaded_files, 1):
                st.write(f"{i}. {f.name}")
        
        st.divider()
        
        # Check if we have processed results in session state
        has_results = 'multi_combined_report_summary' in st.session_state
        
        # Show different UI based on whether results exist
        if not has_results:
            # Add a button to start processing
            if st.button("🚀 Start Batch Processing", type="primary"):
                process_multiple_files(uploaded_files)
                # Store processed file names
                st.session_state.multi_processed_file_names = uploaded_file_names
                st.rerun()
        else:
            # Show results and option to reprocess
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"Results ready! Showing analysis for {len(uploaded_files)} files.")
            with col2:
                if st.button("🔄 Reprocess Files"):
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        if key.startswith('multi_'):
                            del st.session_state[key]
                    st.rerun()
            
            # Display the results
            display_multi_file_results()
    else:
        st.info("Upload multiple PBIX files to get started with batch analysis.")
