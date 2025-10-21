import streamlit as st
import pandas as pd
import io
from src.report import extract_report_metadata

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

def aggrid_table(df, fit_columns=True):
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(groupable=True, enableRowGroup=True, aggFunc="sum", editable=False)
        gb.configure_side_bar()
        gridOptions = gb.build()
        AgGrid(df, gridOptions=gridOptions, fit_columns_on_grid_load=fit_columns)
    except ImportError:
        st.dataframe(df)

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
        slicer_mask = df_visuals.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
        slicers = df_visuals.loc[slicer_mask, 'Visual ID'].nunique()
        non_slicer_visuals = total_unique_visuals - slicers
    else:
        non_slicer_visuals = 0
        slicers = 0

    # Get PBIXRay metrics if model is available
    tables_count = 0
    dax_measures_count = 0
    columns_count = 0
    
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

    # Report summary dataframe with requested columns
    report_name = getattr(uploaded_file, 'name', 'Report').replace('.pbix', '')
    df_report_summary = pd.DataFrame([
        {
            'Report Name': report_name,
            'Pages': summary.get('Total Pages', 0),
            'Visuals': non_slicer_visuals,
            'Slicers': slicers,
            'Static Elements': summary.get('Total Static Elements', 0),
            'Filters': total_filters,
            'Tables': tables_count,
            'DAX Measures': dax_measures_count,
            'Columns': columns_count,
        }
    ])

    # Page summary dataframe (ordered)
    df_pages = pd.DataFrame(pages)
    if not df_pages.empty:
        if not df_visuals.empty and 'Visual ID' in df_visuals.columns:
            visual_ids_by_page = df_visuals.groupby('Page Name')['Visual ID'].nunique()
            slicers_mask = df_visuals.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
            slicers_by_page = df_visuals[slicers_mask].groupby('Page Name')['Visual ID'].nunique()
            non_slicer_by_page = (visual_ids_by_page - slicers_by_page).fillna(0).astype(int)
        else:
            slicers_by_page = pd.Series(dtype=int)
            non_slicer_by_page = pd.Series(dtype=int)

        df_pages = df_pages.copy()
        df_pages['Slicers'] = df_pages['Page Name'].map(slicers_by_page).fillna(0).astype(int)
        df_pages['Visual Count(no slicers)'] = df_pages['Page Name'].map(non_slicer_by_page).fillna(0).astype(int)
        desired_order_pages = ['Page Name', 'All Elements', 'Slicers', 'Static Elements', 'Visual Count(no slicers)', 'Page Filters', 'Groups']
        df_pages_summary = df_pages.reindex(columns=[c for c in desired_order_pages if c in df_pages.columns])
    else:
        df_pages_summary = pd.DataFrame(columns=['Page Name', 'All Elements', 'Slicers', 'Static Elements', 'Visual Count(no slicers)', 'Page Filters', 'Groups'])

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


def process_single_file(uploaded_file):
    model = None
    if PBIXRay is not None:
        model = PBIXRay(uploaded_file)
    else:
        st.warning("pbixray is not installed; showing extractor-based summaries only.")

    # 1. Report Summary (first)
    st.subheader("Report Summary")
    summaries = build_report_summaries(uploaded_file, model)
    if summaries:
        st.dataframe(summaries['report_summary'], hide_index=True, width="stretch")
    else:
        st.info("Could not extract report summary from the PBIX file.")

    # 2. Table Analysis
    if model is not None:
        try:
            if getattr(model, 'power_query', pd.DataFrame()).size:
                st.subheader("Table Analysis")
                st.dataframe(model.power_query)
            else:
                st.info("No Table Analysis found.")
        except Exception:
            st.info("Table Analysis not available.")

    # 3. Relationship Analysis
    if model is not None:
        try:
            if getattr(model, 'relationships', pd.DataFrame()).size:
                st.subheader("Relationships Analysis")
                st.dataframe(model.relationships)
            else:
                st.info("No Relationships found.")
        except Exception:
            st.info("Relationships Analysis not available.")

    # 4. Column Analysis
    if model is not None:
        try:
            st.subheader("Column Analysis")
            st.dataframe(getattr(model, 'statistics', pd.DataFrame()))
        except Exception:
            st.info("Column Analysis not available.")

    # 5. M Parameters
    if model is not None:
        try:
            if getattr(model, 'm_parameters', pd.DataFrame()).size:
                st.subheader("M Parameters")
                st.dataframe(model.m_parameters)
            else:
                st.info("No M Parameters found.")
        except Exception:
            st.info("M Parameters not available.")

    # 6. DAX Tables
    if model is not None:
        try:
            if getattr(model, 'dax_tables', pd.DataFrame()).size:
                st.subheader("DAX Tables")
                st.dataframe(model.dax_tables)
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
                st.dataframe(calculated_cols)
            else:
                st.info("No Calculated Columns found.")
        except Exception as e:
            st.info(f"Calculated Columns not available: {str(e)}")

    # 8. DAX Measures
    if model is not None:
        try:
            if getattr(model, 'dax_measures', pd.DataFrame()).size:
                st.subheader("DAX Measures")
                st.dataframe(model.dax_measures)
            else:
                st.info("No DAX measures found.")
        except Exception:
            st.info("DAX Measures not available.")

    # 9. Page Summary
    if summaries:
        st.subheader("Page Summary")
        st.dataframe(summaries['page_summary'], hide_index=True, width="stretch")

    # 10. Visual Summary
    if summaries:
        st.subheader("Visual Summary")
        st.dataframe(summaries['visual_summary'], hide_index=True, width="stretch")

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
    
    st.download_button(
        label="📥 Download Complete Analysis (Excel)",
        data=export_buffer.getvalue(),
        file_name=f"{getattr(uploaded_file, 'name', 'analysis').replace('.pbix', '')}_complete_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Main app
st.title("📊 PBIX Analyser for Reckitt")

uploaded_file = st.file_uploader("📁 Upload a PBIX file", type="pbix")

if uploaded_file:
    process_single_file(uploaded_file)
else:
    st.info("Upload a PBIX file to get started.")
