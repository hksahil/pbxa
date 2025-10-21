import streamlit as st
import pandas as pd
from src.report import extract_report_metadata
from src.utils import filter_dataframe_by_text
import io
import json
import os

# Page configuration
st.set_page_config(
    page_title="More Detailed Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for processing mode selection
st.sidebar.title("⚙️ Processing Mode")
st.sidebar.markdown("---")

processing_mode = st.sidebar.radio(
    "Select how you want to process PBIX files:",
    options=[
        "📄 Single File Analysis",
        "📚 Multiple Files Comparison"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Processing Modes:
- **Single File**: Upload and analyze one PBIX file in detail
- **Multiple Files**: Upload and compare multiple PBIX files side-by-side
""")

# Title and description
st.title("📊 More Detailed Analysis")

def process_single_pbix(uploaded_file, file_name="Report"):
    """Process a single PBIX file and return the data"""
    try:
        with st.spinner(f'🔄 Processing {file_name}...'):
            report_data = extract_report_metadata(uploaded_file)
        return report_data
    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {str(e)}")
        return None


def calculate_report_metrics(report_data):
    #st.write(report_data)
    """Calculate advanced metrics for a report including complexity score"""
    summary = report_data.get("summary", {})
    visuals = report_data.get("visuals", [])
    pages = report_data.get("pages", [])
    
    df_visuals = pd.DataFrame(visuals)
    
    # Count measures
    measures_count = len(df_visuals[df_visuals['Is Measure'] == 'Yes']) if not df_visuals.empty else 0
    
    # Count unique tables (from Field Query Name which has format like "Sum(TableName[Column])" or "TableName.Column")
    unique_tables = set()
    if not df_visuals.empty:
        for query_name in df_visuals['Field Query Name'].dropna():
            query_name = str(query_name).strip()
            
            # Handle format: TableName[Column] or Sum(TableName[Column])
            if '[' in query_name:
                # Extract table name before the bracket
                # Remove any aggregation functions first
                clean_name = query_name
                if '(' in clean_name:
                    # Extract content between ( and )
                    start = clean_name.find('(')
                    end = clean_name.rfind(')')
                    if start != -1 and end != -1:
                        clean_name = clean_name[start+1:end]
                
                # Now extract table name before [
                if '[' in clean_name:
                    table = clean_name.split('[')[0].strip()
                    if table and not table.startswith('_'):  # Exclude system tables
                        unique_tables.add(table)
            
            # Handle format: TableName.Column
            elif '.' in query_name and not query_name.startswith('.'):
                # Split by the last dot to separate table from column
                parts = query_name.rsplit('.', 1)
                if len(parts) == 2:
                    table = parts[0].strip()
                    # Remove any aggregation function prefix
                    if '(' in table:
                        table = table.split('(')[-1]
                    if table and not table.startswith('_'):
                        unique_tables.add(table)
    
    tables_count = len(unique_tables)
    
    # Count filters
    total_filters = 0
    for page in pages:
        page_filters = page.get('Page Filters', '')
        if page_filters and page_filters != "None":
            total_filters += len(page_filters.split(' | '))
    
    unique_visual_filters = df_visuals[['Visual ID', 'Visual Filters']].drop_duplicates() if not df_visuals.empty else pd.DataFrame()
    for _, visual in unique_visual_filters.iterrows():
        visual_filters = visual.get('Visual Filters', '')
        if visual_filters:
            total_filters += len(visual_filters.split(' | '))
    
    # Calculate Complexity Score
    # Formula: (Pages * 10) + (Visuals * 5) + (Fields * 2) + (Measures * 3) + (Tables * 8) + (Filters * 4)
    # Weighted to reflect complexity impact
    total_pages = summary.get("Total Pages", 0)
    total_visuals = summary.get("Total Visuals", 0)
    total_fields = len(visuals)
    
    complexity_score = (
        (total_pages * 10) +
        (total_visuals * 5) +
        (total_fields * 2) +
        (measures_count * 3) +
        (tables_count * 8) +
        (total_filters * 4)
    )
    
    # Determine complexity level
    if complexity_score < 100:
        complexity_level = "Low"
    elif complexity_score < 500:
        complexity_level = "Medium"
    elif complexity_score < 1000:
        complexity_level = "High"
    else:
        complexity_level = "Very High"
    
    return {
        "measures_count": measures_count,
        "tables_count": tables_count,
        "total_filters": total_filters,
        "complexity_score": complexity_score,
        "complexity_level": complexity_level,
        "unique_tables": list(unique_tables)  # For debugging
    }

def display_report_data(report_data, report_name="Report"):
    """Display report data in tabs"""
    
    if not report_data or not report_data.get("visuals"):
        st.warning(f"⚠️ No visuals with data projections found in {report_name}.")
        return
    
    # Add global text filter
    st.markdown("### Filter All Data")
    search_text = st.text_input(
        "Search across all data in this report:",
        placeholder="Enter text to filter all rows...",
        help="This will filter all data tables below based on your search text (case-insensitive)",
        key=f"global_search_{report_name}"
    )
    
    if search_text:
        st.info(f"🔎 Filtering all data for: '{search_text}'")
    
    st.divider()
    
    summary = report_data.get("summary", {})
    pages = report_data.get("pages", [])
    visuals = report_data.get("visuals", [])
    
    # Convert to DataFrames
    df_pages = pd.DataFrame(pages)
    df_visuals = pd.DataFrame(visuals)
    
    # ========== TABS FOR DIFFERENT VIEWS ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Page Overview", "📊 Visual Details", "🔍 Filters", "📥 Export Data"])
    
    # TAB 1: PAGE OVERVIEW
    with tab1:
        st.subheader("📄 Pages Summary")
        
        if not df_pages.empty:
            # Compute slicer counts and adjust visual counts to exclude slicers
            visual_ids_by_page = df_visuals.groupby('Page Name')['Visual ID'].nunique() if 'Visual ID' in df_visuals.columns else pd.Series(dtype=int)
            slicers_mask = df_visuals.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
            slicers_by_page = df_visuals[slicers_mask].groupby('Page Name')['Visual ID'].nunique() if not df_visuals.empty else pd.Series(dtype=int)
            non_slicer_by_page = (visual_ids_by_page - slicers_by_page).fillna(0).astype(int)

            # Add/override columns in df_pages for display
            df_pages = df_pages.copy()
            df_pages['Slicers'] = df_pages['Page Name'].map(slicers_by_page).fillna(0).astype(int)
            df_pages['Visual Count'] = df_pages['Page Name'].map(non_slicer_by_page).fillna(0).astype(int)

            # Reorder columns per requested order (keep Page Name first)
            desired_order = ['Page Name', 'All Elements', 'Slicers', 'Static Elements', 'Visual Count', 'Page Filters', 'Groups']
            display_df = df_pages.reindex(columns=[c for c in desired_order if c in df_pages.columns])
            
            # Apply text filter
            filtered_display_df = filter_dataframe_by_text(display_df, search_text)

            st.dataframe(
                filtered_display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Page Name": st.column_config.TextColumn("Page Name", width="medium"),
                    "All Elements": st.column_config.NumberColumn("All Elements", width="small"),
                    "Slicers": st.column_config.NumberColumn("Slicers", width="small"),
                    "Static Elements": st.column_config.NumberColumn("Static Elements", width="small"),
                    "Visual Count": st.column_config.NumberColumn("Visual Count(no slicers)", width="small"),
                    "Page Filters": st.column_config.TextColumn("Page Filters", width="large"),
                    "Groups": st.column_config.NumberColumn("Groups", width="small"),
                }
            )
            
            if search_text and len(filtered_display_df) < len(display_df):
                st.caption(f"Showing {len(filtered_display_df)} of {len(display_df)} pages")
            
            st.divider()
            st.subheader("Page Drill-Down")
            
            selected_page = st.selectbox(
                "Select a page to view its visuals:",
                options=df_pages['Page Name'].unique(),
                key=f"page_select_{report_name}"
            )
            
            if selected_page:
                page_visuals = df_visuals[df_visuals['Page Name'] == selected_page]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Visuals on Page", page_visuals['Visual ID'].nunique())
                with col2:
                    st.metric("Total Fields", len(page_visuals))
                with col3:
                    measures_count = len(page_visuals[page_visuals['Is Measure'] == 'Yes'])
                    st.metric("Measures Used", measures_count)
                
                st.markdown("**Visuals on this page:**")
                page_visual_types = page_visuals.groupby('Visual Type')['Visual ID'].nunique().reset_index()
                page_visual_types.columns = ['Visual Type', 'Count']
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(page_visual_types, width="stretch", hide_index=True)
                with col2:
                    st.bar_chart(page_visual_types.set_index('Visual Type'))
        else:
            st.info("No page data available")
    
    # TAB 2: VISUAL DETAILS
    with tab2:
        st.subheader("📊 Visual Details")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            selected_pages = st.multiselect(
                "Filter by Page",
                options=df_visuals['Page Name'].unique(),
                default=df_visuals['Page Name'].unique(),
                key=f"pages_{report_name}"
            )
        
        with col2:
            selected_visual_types = st.multiselect(
                "Filter by Visual Type",
                options=df_visuals['Visual Type'].unique(),
                default=df_visuals['Visual Type'].unique(),
                key=f"types_{report_name}"
            )
        
        with col3:
            selected_titles = st.multiselect(
                "Filter by Title",
                options=df_visuals['Visual Title'].unique(),
                default=df_visuals['Visual Title'].unique(),
                key=f"titles_{report_name}"
            )
        
        with col4:
            measure_filter = st.selectbox(
                "Filter by Type",
                options=["All", "Measures Only", "Columns Only"],
                key=f"measure_{report_name}"
            )
        
        with col5:
            has_filters = st.selectbox(
                "Has Visual Filters",
                options=["All", "With Filters", "Without Filters"],
                key=f"filters_{report_name}"
            )
        
        filtered_df = df_visuals[
            (df_visuals['Page Name'].isin(selected_pages)) &
            (df_visuals['Visual Type'].isin(selected_visual_types)) &
            (df_visuals['Visual Title'].isin(selected_titles))
        ]
        
        if measure_filter == "Measures Only":
            filtered_df = filtered_df[filtered_df['Is Measure'] == 'Yes']
        elif measure_filter == "Columns Only":
            filtered_df = filtered_df[filtered_df['Is Measure'] == 'No']
        
        if has_filters == "With Filters":
            filtered_df = filtered_df[filtered_df['Visual Filters'].str.len() > 0]
        elif has_filters == "Without Filters":
            filtered_df = filtered_df[filtered_df['Visual Filters'].str.len() == 0]
        
        # Apply global text search filter
        filtered_df_before_text = filtered_df.copy()
        filtered_df = filter_dataframe_by_text(filtered_df, search_text)
        
        st.divider()
        st.markdown(f"**Showing {len(filtered_df)} records**")
        if search_text and len(filtered_df) < len(filtered_df_before_text):
            st.caption(f"Text filter reduced from {len(filtered_df_before_text)} to {len(filtered_df)} records")
        
        # Add Suspect for change? flag based on keywords across Field/Query/Title
        def compute_suspect_flag(df: pd.DataFrame) -> pd.DataFrame:
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
            df['Suspect for change?'] = df.apply(row_flag, axis=1)
            return df

        filtered_df = compute_suspect_flag(filtered_df)
        
        if st.checkbox("Group by Visual", value=True, key=f"group_{report_name}"):
            unique_visuals = filtered_df.groupby(['Page Name', 'Visual ID', 'Visual Title', 'Visual Type']).size().reset_index(name='Field Count')
            
            for idx, visual in unique_visuals.iterrows():
                title_display = f"{visual['Visual Title']} ({visual['Visual Type']})" 
                with st.expander(f"🔹 {title_display} on {visual['Page Name']} ({visual['Field Count']} fields)"):
                    visual_data = filtered_df[
                        (filtered_df['Page Name'] == visual['Page Name']) &
                        (filtered_df['Visual ID'] == visual['Visual ID'])
                    ]
                    
                    st.dataframe(
                        visual_data[[
                            'Field Display Name', 'Field Query Name', 'Field Type', 
                            'Field Format', 'Is Measure', 'Aggregation', 'Projection Type',
                            'Suspect for change?'
                        ]],
                        width="stretch",
                        hide_index=True
                    )
        else:
            st.dataframe(
                filtered_df.drop(columns=['Visual ID', 'Visual Filters'], errors='ignore'),
                width="stretch",
                hide_index=True,
                column_config={
                    "Page Name": st.column_config.TextColumn("Page", width="small"),
                    "Visual Title": st.column_config.TextColumn("Title", width="medium"),
                    "Visual Type": st.column_config.TextColumn("Visual Type", width="small"),
                    "Field Display Name": st.column_config.TextColumn("Field", width="medium"),
                    "Field Type": st.column_config.TextColumn("Type", width="small"),
                    "Is Measure": st.column_config.TextColumn("Measure?", width="small"),
                    "Suspect for change?": st.column_config.TextColumn("Suspect for change?", width="small"),
                }
            )
        
        st.divider()
        st.subheader("📊 Visual Type Distribution")
        visual_type_counts = filtered_df.groupby('Visual Type')['Visual ID'].nunique()
        st.bar_chart(visual_type_counts)
    
    # TAB 3: FILTERS
    with tab3:
        st.subheader("Filter Analysis")
        
        filters_data = []
        
        for _, page in df_pages.iterrows():
            page_filters_str = page['Page Filters']
            if page_filters_str and page_filters_str != "None":
                filter_items = page_filters_str.split(' | ')
                for filter_item in filter_items:
                    if ':' in filter_item:
                        field_part, condition_part = filter_item.split(':', 1)
                        if '.' in field_part:
                            table, column = field_part.rsplit('.', 1)
                            filters_data.append({
                                'Filter Level': 'Page',
                                'Page Name': page['Page Name'],
                                'Visual Title': 'N/A',
                                'Visual Type': 'N/A',
                                'Table': table.strip(),
                                'Column': column.strip(),
                                'Condition': condition_part.strip()
                            })
        
        unique_visual_filters = df_visuals[['Page Name', 'Visual ID', 'Visual Title', 'Visual Type', 'Visual Filters']].drop_duplicates()
        
        for _, visual in unique_visual_filters.iterrows():
            visual_filters_str = visual['Visual Filters']
            if visual_filters_str:
                filter_items = visual_filters_str.split(' | ')
                for filter_item in filter_items:
                    if ':' in filter_item:
                        field_part, condition_part = filter_item.split(':', 1)
                        if '.' in field_part:
                            table, column = field_part.rsplit('.', 1)
                            filters_data.append({
                                'Filter Level': 'Visual',
                                'Page Name': visual['Page Name'],
                                'Visual Title': visual['Visual Title'],
                                'Visual Type': visual['Visual Type'],
                                'Table': table.strip(),
                                'Column': column.strip(),
                                'Condition': condition_part.strip()
                            })
        
        if filters_data:
            df_filters = pd.DataFrame(filters_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Filters", len(df_filters))
            with col2:
                page_filters_count = len(df_filters[df_filters['Filter Level'] == 'Page'])
                st.metric("Page Filters", page_filters_count)
            with col3:
                visual_filters_count = len(df_filters[df_filters['Filter Level'] == 'Visual'])
                st.metric("Visual Filters", visual_filters_count)
            with col4:
                unique_tables = df_filters['Table'].nunique()
                st.metric("Tables Filtered", unique_tables)
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_level_options = st.multiselect(
                    "Filter Level",
                    options=df_filters['Filter Level'].unique(),
                    default=df_filters['Filter Level'].unique(),
                    key=f"filter_level_{report_name}"
                )
            with col2:
                page_filter_options = st.multiselect(
                    "Page",
                    options=df_filters['Page Name'].unique(),
                    default=df_filters['Page Name'].unique(),
                    key=f"page_filter_{report_name}"
                )
            with col3:
                table_filter_options = st.multiselect(
                    "Table",
                    options=sorted(df_filters['Table'].unique()),
                    default=df_filters['Table'].unique(),
                    key=f"table_filter_{report_name}"
                )
            
            filtered_filters_df = df_filters[
                (df_filters['Filter Level'].isin(filter_level_options)) &
                (df_filters['Page Name'].isin(page_filter_options)) &
                (df_filters['Table'].isin(table_filter_options))
            ]
            
            # Apply global text search filter
            filtered_filters_df_before_text = filtered_filters_df.copy()
            filtered_filters_df = filter_dataframe_by_text(filtered_filters_df, search_text)
            
            st.divider()
            st.markdown(f"**Showing {len(filtered_filters_df)} filters**")
            if search_text and len(filtered_filters_df) < len(filtered_filters_df_before_text):
                st.caption(f"Text filter reduced from {len(filtered_filters_df_before_text)} to {len(filtered_filters_df)} filters")
            
            st.dataframe(
                filtered_filters_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Filter Level": st.column_config.TextColumn("Level", width="small"),
                    "Page Name": st.column_config.TextColumn("Page", width="medium"),
                    "Visual Title": st.column_config.TextColumn("Title", width="medium"),
                    "Visual Type": st.column_config.TextColumn("Visual", width="small"),
                    "Table": st.column_config.TextColumn("Table", width="medium"),
                    "Column": st.column_config.TextColumn("Column", width="medium"),
                    "Condition": st.column_config.TextColumn("Condition", width="large")
                }
            )
            
            st.divider()
            st.subheader("📊 Filters by Table")
            filters_by_table = filtered_filters_df['Table'].value_counts()
            st.bar_chart(filters_by_table)
            
        else:
            st.info("No filters found in the report.")
    
    # TAB 4: EXPORT DATA
    with tab4:
        st.subheader("💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Page Summary")
            csv_pages = df_pages.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Page Summary (CSV)",
                data=csv_pages,
                file_name=f"{report_name}_pages_summary.csv",
                mime="text/csv",
                key=f"export_pages_{report_name}"
            )
        
        with col2:
            st.markdown("### 📊 Visual Details")
            csv_visuals = df_visuals.drop(columns=['Visual ID', 'Visual Filters'], errors='ignore').to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Visual Details (CSV)",
                data=csv_visuals,
                file_name=f"{report_name}_visuals_details.csv",
                mime="text/csv",
                key=f"export_visuals_{report_name}"
            )
        
        st.divider()
        
        if filters_data:
            st.markdown("### 🔍 Filters")
            csv_filters = df_filters.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filters (CSV)",
                data=csv_filters,
                file_name=f"{report_name}_filters.csv",
                mime="text/csv",
                key=f"export_filters_{report_name}"
            )
            
            st.divider()
        
        st.markdown("### 📑 Complete Report (Excel)")
        output = io.BytesIO()

        # Build Sheet 1: Comparison Summary (single report)
        # Compute slicers and non-slicer visuals
        if not df_visuals.empty and 'Visual ID' in df_visuals.columns:
            total_unique_visuals = df_visuals['Visual ID'].nunique()
            slicer_mask_sum = df_visuals.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
            slicers_count_sum = df_visuals.loc[slicer_mask_sum, 'Visual ID'].nunique()
            non_slicer_visuals_sum = total_unique_visuals - slicers_count_sum
        else:
            slicers_count_sum = 0
            non_slicer_visuals_sum = 0

        # Compute measures, tables, filters, complexity using helper
        metrics_sum = calculate_report_metrics({"summary": summary, "visuals": visuals, "pages": pages})
        complexity_emojis_sum = {
            "Low": "🟢 Low",
            "Medium": "🟡 Medium",
            "High": "🟠 High",
            "Very High": "🔴 Very High"
        }
        df_summary_export = pd.DataFrame([
            {
                "Report Name": report_name,
                "Pages": summary.get("Total Pages", 0),
                "Visuals": non_slicer_visuals_sum,
                "Slicers": slicers_count_sum,
                "Static Elements": summary.get("Total Static Elements", 0),
                "Fields": len(visuals),
                "Measures": metrics_sum["measures_count"],
                "Tables": metrics_sum["tables_count"],
                "Filters": metrics_sum["total_filters"],
                "Complexity Score": metrics_sum["complexity_score"],
                "Complexity Level": complexity_emojis_sum.get(metrics_sum["complexity_level"], metrics_sum["complexity_level"]) 
            }
        ])

        # Build Sheet 2: Pages Summary with required columns
        df_pages_sheet = pd.DataFrame(pages)
        if not df_pages_sheet.empty:
            # compute slicers and non-slicer counts
            if not df_visuals.empty and 'Visual ID' in df_visuals.columns:
                visual_ids_by_page_exp = df_visuals.groupby('Page Name')['Visual ID'].nunique()
                slicers_by_page_exp = df_visuals[df_visuals['Visual Type'].astype(str).str.lower() == 'slicer'].groupby('Page Name')['Visual ID'].nunique()
                non_slicer_by_page_exp = (visual_ids_by_page_exp - slicers_by_page_exp).fillna(0).astype(int)
            else:
                slicers_by_page_exp = pd.Series(dtype=int)
                non_slicer_by_page_exp = pd.Series(dtype=int)

            df_pages_sheet = df_pages_sheet.copy()
            df_pages_sheet['Slicers'] = df_pages_sheet['Page Name'].map(slicers_by_page_exp).fillna(0).astype(int)
            df_pages_sheet['Visual Count'] = df_pages_sheet['Page Name'].map(non_slicer_by_page_exp).fillna(0).astype(int)
            desired_pages_cols = ['Page Name', 'All Elements', 'Slicers', 'Static Elements', 'Visual Count', 'Page Filters', 'Groups']
            df_pages_export_excel = df_pages_sheet.reindex(columns=[c for c in desired_pages_cols if c in df_pages_sheet.columns])
        else:
            df_pages_export_excel = df_pages_sheet

        # Build Sheet 3: Visuals Summary with required columns and suspect flag
        def compute_suspect_flag_for_export(df: pd.DataFrame) -> pd.DataFrame:
            keywords = [
                "eh",
                "essential home",
                "essential homes",
                "bu",
                "functional",
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

        df_visuals_sheet = pd.DataFrame(visuals)
        if not df_visuals_sheet.empty:
            df_visuals_sheet = compute_suspect_flag_for_export(df_visuals_sheet)
            # Map to requested headers
            df_visuals_export_excel = pd.DataFrame()
            df_visuals_export_excel['Page'] = df_visuals_sheet.get('Page Name', pd.Series(dtype=str))
            df_visuals_export_excel['Title'] = df_visuals_sheet.get('Visual Title', pd.Series(dtype=str))
            df_visuals_export_excel['Visual Type'] = df_visuals_sheet.get('Visual Type', pd.Series(dtype=str))
            df_visuals_export_excel['Field'] = df_visuals_sheet.get('Field Display Name', pd.Series(dtype=str))
            df_visuals_export_excel['Field Query Name'] = df_visuals_sheet.get('Field Query Name', pd.Series(dtype=str))
            df_visuals_export_excel['Type'] = df_visuals_sheet.get('Field Type', pd.Series(dtype=str))
            df_visuals_export_excel['Field Format'] = df_visuals_sheet.get('Field Format', pd.Series(dtype=str))
            df_visuals_export_excel['Measure?'] = df_visuals_sheet.get('Is Measure', pd.Series(dtype=str))
            # Keep user's requested labels (including their spellings)
            df_visuals_export_excel['Argregation'] = df_visuals_sheet.get('Aggregation', pd.Series(dtype=str))
            df_visuals_export_excel['Proiection Type'] = df_visuals_sheet.get('Projection Type', pd.Series(dtype=str))
            df_visuals_export_excel['Active'] = df_visuals_sheet.get('Active', pd.Series(dtype=str))
            df_visuals_export_excel['Suspect for change?'] = df_visuals_sheet.get('Suspect for change?', pd.Series(dtype=str))
        else:
            df_visuals_export_excel = df_visuals_sheet

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_summary_export.to_excel(writer, index=False, sheet_name='Summary')
            df_pages_export_excel.to_excel(writer, index=False, sheet_name='Pages')
            df_visuals_export_excel.to_excel(writer, index=False, sheet_name='Visual Details')

        excel_data = output.getvalue()

        st.download_button(
            label="📥 Download Complete Report (Excel)",
            data=excel_data,
            file_name=f"{report_name}_complete_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_excel_{report_name}"
        )

# ========== MAIN APPLICATION LOGIC ==========

if processing_mode == "📄 Single File Analysis":
    st.markdown("""
    Upload a single Power BI (.pbix) file to extract comprehensive metadata including:
    - **Report Summary**: Pages and visuals count
    - **Page Details**: Visual count and filters per page  
    - **Visual Details**: Fields, measures, data types, titles, and filters
    - **Filter Analysis**: Dedicated view of all page and visual-level filters
    """)
    
    uploaded_file = st.file_uploader("Upload a Power BI (.pbix) file", type=['pbix'])
    
    if uploaded_file is not None:
        report_data = process_single_pbix(uploaded_file, uploaded_file.name)
        if report_data:
            # Single-file Comparison Summary (one row)
            st.header("📊 Comparison Summary")

            summary = report_data.get("summary", {})
            visuals = report_data.get("visuals", [])

            # Calculate metrics
            metrics = calculate_report_metrics(report_data)

            # Calculate slicer and non-slicer visual counts (by unique Visual ID)
            df_visuals_tmp = pd.DataFrame(visuals) if visuals else pd.DataFrame()
            if not df_visuals_tmp.empty and 'Visual ID' in df_visuals_tmp.columns:
                total_unique_visuals = df_visuals_tmp['Visual ID'].nunique()
                slicer_mask = df_visuals_tmp.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
                slicers_count = df_visuals_tmp.loc[slicer_mask, 'Visual ID'].nunique()
                non_slicer_visuals_count = total_unique_visuals - slicers_count
            else:
                total_unique_visuals = 0
                slicers_count = 0
                non_slicer_visuals_count = 0

            complexity_emojis = {
                "Low": "🟢 Low",
                "Medium": "🟡 Medium",
                "High": "🟠 High",
                "Very High": "🔴 Very High"
            }

            report_name = uploaded_file.name.replace('.pbix', '')
            df_comparison = pd.DataFrame([
                {
                    "Report Name": report_name,
                    "Total Pages": summary.get("Total Pages", 0),
                    "Total Visuals": non_slicer_visuals_count,
                    "Slicers": slicers_count,
                    "Static Elements": summary.get("Total Static Elements", 0),
                    "Total Fields": len(visuals),
                    "Measures": metrics["measures_count"],
                    "Tables": metrics["tables_count"],
                    "Total Filters": metrics["total_filters"],
                    "Complexity Score": metrics["complexity_score"],
                    "Complexity": complexity_emojis.get(metrics["complexity_level"], metrics["complexity_level"])
                }
            ])

            st.dataframe(
                df_comparison,
                width="stretch",
                hide_index=True,
                column_config={
                    "Report Name": st.column_config.TextColumn("Report Name", width="large"),
                    "Total Pages": st.column_config.NumberColumn("Pages", width="small"),
                    "Total Visuals": st.column_config.NumberColumn("Visuals", width="small"),
                    "Slicers": st.column_config.NumberColumn("Slicers", width="small"),
                    "Static Elements": st.column_config.NumberColumn("Static Elements", width="small"),
                    "Total Fields": st.column_config.NumberColumn("Fields", width="small"),
                    "Measures": st.column_config.NumberColumn("Measures", width="small"),
                    "Tables": st.column_config.NumberColumn("Tables", width="small"),
                    "Total Filters": st.column_config.NumberColumn("Filters", width="small"),
                    "Complexity Score": st.column_config.NumberColumn("Complexity Score", width="small"),
                    "Complexity": st.column_config.TextColumn("Complexity Level", width="small"),
                }
            )

            st.markdown("**Complexity Level Legend:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🟢 **Low** (< 100)")
            with col2:
                st.markdown("🟡 **Medium** (100-499)")
            with col3:
                st.markdown("🟠 **High** (500-999)")
            with col4:
                st.markdown("🔴 **Very High** (≥ 1000)")

            st.divider()
            csv_comparison = df_comparison.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Comparison Summary (CSV)",
                data=csv_comparison,
                file_name=f"{report_name}_comparison_summary.csv",
                mime="text/csv",
            )

            st.divider()
            # Export Automated Analysis (Excel with 3 sheets)
            export_buffer = io.BytesIO()

            # Build Pages Summary like UI (include slicers and non-slicer counts)
            df_pages = pd.DataFrame(report_data.get("pages", []))
            df_visuals_full = pd.DataFrame(visuals)

            # Compute slicers and non-slicer visual counts per page
            if not df_visuals_full.empty and 'Visual ID' in df_visuals_full.columns:
                visual_ids_by_page = df_visuals_full.groupby('Page Name')['Visual ID'].nunique()
                slicers_mask = df_visuals_full.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
                slicers_by_page = df_visuals_full[slicers_mask].groupby('Page Name')['Visual ID'].nunique()
                non_slicer_by_page = (visual_ids_by_page - slicers_by_page).fillna(0).astype(int)
            else:
                slicers_by_page = pd.Series(dtype=int)
                non_slicer_by_page = pd.Series(dtype=int)

            if not df_pages.empty:
                df_pages = df_pages.copy()
                df_pages['Slicers'] = df_pages['Page Name'].map(slicers_by_page).fillna(0).astype(int)
                df_pages['Visual Count'] = df_pages['Page Name'].map(non_slicer_by_page).fillna(0).astype(int)
                desired_order = ['Page Name', 'All Elements', 'Slicers', 'Static Elements', 'Visual Count', 'Page Filters', 'Groups']
                df_pages_export = df_pages.reindex(columns=[c for c in desired_order if c in df_pages.columns])
            else:
                df_pages_export = df_pages

            # Build Visuals Summary like Visual Details export with Suspect flag
            def compute_suspect_flag_export(df: pd.DataFrame) -> pd.DataFrame:
                keywords = [
                    "eh",
                    "essential home",
                    "essential homes",
                    "bu",
                    "functional",
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

            if not df_visuals_full.empty:
                df_visuals_export = compute_suspect_flag_export(df_visuals_full)
                # Export the same columns as the ungrouped Visual Details matrix in a fixed order
                desired_visual_cols = [
                    'Page Name',
                    'Visual Title',
                    'Visual Type',
                    'Field Display Name',
                    'Field Query Name',
                    'Field Type',
                    'Field Format',
                    'Is Measure',
                    'Aggregation',
                    'Projection Type',
                    'Active',
                    'Suspect for change?'
                ]
                # Keep any available columns in that order
                df_visuals_export = df_visuals_export.reindex(columns=[c for c in desired_visual_cols if c in df_visuals_export.columns])
            else:
                df_visuals_export = df_visuals_full

            with pd.ExcelWriter(export_buffer, engine='openpyxl') as writer:
                # Sheet 1: Comparison Summary
                df_comparison.to_excel(writer, index=False, sheet_name='Comparison Summary')
                # Sheet 2: Pages Summary
                df_pages_export.to_excel(writer, index=False, sheet_name='Pages Summary')
                # Sheet 3: Visuals Summary
                df_visuals_export.to_excel(writer, index=False, sheet_name='Visuals Summary')

            st.download_button(
                label="📤 Export Automated Analysis (Excel)",
                data=export_buffer.getvalue(),
                file_name=f"{report_name}_automated_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.divider()
            display_report_data(report_data, report_name)
    else:
        with st.expander("ℹ️ How to use Single File Analysis"):
            st.markdown("""
            ### Steps:
            1. Click on **"Browse files"** above
            2. Select a Power BI (.pbix) file from your computer
            3. Wait for processing to complete
            4. Explore the results in four tabs:
               - **Page Overview**: Summary of pages and their visuals
               - **Visual Details**: Detailed field-level information
               - **Filters**: All filters in a dedicated view
               - **Export Data**: Download results as CSV or Excel
            
            ### What's Extracted:
            - Visual titles and types
            - Fields and measures with data types
            - Page-level and visual-level filters
            - Aggregations and projections
            - Format information
            """)

elif processing_mode == "📚 Multiple Files Comparison":
    st.markdown("""
    Upload multiple Power BI (.pbix) files to compare and analyze them side-by-side.
    
    **Perfect for:**
    - Comparing development vs production reports
    - Analyzing report variations across different teams
    - Batch processing multiple reports
    - Cross-report analysis
    """)
    
    uploaded_files = st.file_uploader(
        "Upload Power BI (.pbix) files", 
        type=['pbix'], 
        accept_multiple_files=True,
        help="You can select multiple files at once by holding Ctrl (Windows) or Cmd (Mac)"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Process all files
        all_reports_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            report_name = uploaded_file.name.replace('.pbix', '')
            status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            report_data = process_single_pbix(uploaded_file, uploaded_file.name)
            if report_data:
                all_reports_data[report_name] = report_data
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        if all_reports_data:
            # Comparison Summary
            st.header("📊 Comparison Summary")
            
            comparison_data = []
            for report_name, report_data in all_reports_data.items():
                summary = report_data.get("summary", {})
                visuals = report_data.get("visuals", [])
                
                # Calculate metrics using helper function
                metrics = calculate_report_metrics(report_data)
                
                # Calculate slicer and non-slicer visual counts (by unique Visual ID)
                df_visuals_tmp = pd.DataFrame(visuals) if visuals else pd.DataFrame()
                if not df_visuals_tmp.empty and 'Visual ID' in df_visuals_tmp.columns:
                    total_unique_visuals = df_visuals_tmp['Visual ID'].nunique()
                    slicer_mask = df_visuals_tmp.get('Visual Type', pd.Series(dtype=str)).astype(str).str.lower() == 'slicer'
                    slicers_count = df_visuals_tmp.loc[slicer_mask, 'Visual ID'].nunique()
                    non_slicer_visuals_count = total_unique_visuals - slicers_count
                else:
                    total_unique_visuals = 0
                    slicers_count = 0
                    non_slicer_visuals_count = 0
                
                # Add emoji to complexity level
                complexity_emojis = {
                    "Low": "🟢 Low",
                    "Medium": "🟡 Medium",
                    "High": "🟠 High",
                    "Very High": "🔴 Very High"
                }
                
                comparison_data.append({
                    "Report Name": report_name,
                    "Total Pages": summary.get("Total Pages", 0),
                    # Visuals column should exclude slicers
                    "Total Visuals": non_slicer_visuals_count,
                    "Slicers": slicers_count,
                    "Static Elements": summary.get("Total Static Elements", 0),
                    "Total Fields": len(visuals),
                    "Measures": metrics["measures_count"],
                    "Tables": metrics["tables_count"],
                    "Total Filters": metrics["total_filters"],
                    "Complexity Score": metrics["complexity_score"],
                    "Complexity": complexity_emojis.get(metrics["complexity_level"], metrics["complexity_level"])
                })
            
# Replace the comparison summary display section (around line 543-578)
            df_comparison = pd.DataFrame(comparison_data)
            
            st.dataframe(
                df_comparison, 
                width="stretch", 
                hide_index=True,
                column_config={
                    "Report Name": st.column_config.TextColumn("Report Name", width="large"),
                    "Total Pages": st.column_config.NumberColumn("Pages", width="small"),
                    "Total Visuals": st.column_config.NumberColumn("Visuals", width="small"),
                    "Slicers": st.column_config.NumberColumn("Slicers", width="small"),
                    "Static Elements": st.column_config.NumberColumn("Static Elements", width="small"),
                    "Total Fields": st.column_config.NumberColumn("Fields", width="small"),
                    "Measures": st.column_config.NumberColumn("Measures", width="small"),
                    "Tables": st.column_config.NumberColumn("Tables", width="small"),
                    "Total Filters": st.column_config.NumberColumn("Filters", width="small"),
                    "Complexity Score": st.column_config.NumberColumn("Complexity Score", width="small"),
                    "Complexity": st.column_config.TextColumn("Complexity Level", width="small"),
                }
            )
            
            # Add visual complexity indicators
            st.markdown("**Complexity Level Legend:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🟢 **Low** (< 100)")
            with col2:
                st.markdown("🟡 **Medium** (100-499)")
            with col3:
                st.markdown("🟠 **High** (500-999)")
            with col4:
                st.markdown("🔴 **Very High** (≥ 1000)")
            
            # Export comparison summary
            st.divider()
            csv_comparison = df_comparison.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Comparison Summary (CSV)",
                data=csv_comparison,
                file_name="pbix_comparison_summary.csv",
                mime="text/csv",
            )
            
            st.divider()

            # Optional validation using demo-layout.json if present
            with st.expander("🧪 Validation: demo-layout.json (page totals)"):
                demo_path = os.path.join(os.path.dirname(__file__), "demo-layout.json")
                if os.path.exists(demo_path) and os.path.getsize(demo_path) > 0:
                    try:
                        with open(demo_path, "r", encoding="utf-8") as f:
                            demo_layout = json.load(f)

                        sections = demo_layout.get("sections", [])
                        validation_rows = []
                        for section in sections:
                            page_name = section.get("displayName") or section.get("name") or "Unnamed Page"
                            visual_containers = section.get("visualContainers", [])

                            # Count group containers
                            group_count = 0
                            for vc in visual_containers:
                                cfg_str = vc.get("config", "{}")
                                try:
                                    cfg = json.loads(cfg_str)
                                except Exception:
                                    cfg = {}
                                if "singleVisualGroup" in cfg:
                                    group_count += 1

                            # Detect background image at page level
                            bg_images = 0
                            section_cfg_str = section.get("config", "{}")
                            try:
                                section_cfg = json.loads(section_cfg_str)
                                obj_cfg = section_cfg.get("objects", {})
                                bg_cfg = obj_cfg.get("background", [])
                                if isinstance(bg_cfg, list):
                                    for bg in bg_cfg:
                                        props = bg.get("properties", {})
                                        if props.get("image") or props.get("imageUrl"):
                                            bg_images = 1
                                            break
                            except Exception:
                                pass

                            all_elements = len(visual_containers) + bg_images

                            validation_rows.append({
                                "Page Name": page_name,
                                "All Elements (demo)": all_elements,
                                "Containers": len(visual_containers),
                                "Groups": group_count,
                                "Background Images": bg_images,
                            })

                        if validation_rows:
                            st.dataframe(pd.DataFrame(validation_rows), hide_index=True, width="stretch")
                        else:
                            st.info("No sections found in demo-layout.json")
                    except Exception as e:
                        st.error(f"Failed to parse demo-layout.json: {e}")
                else:
                    st.info("demo-layout.json not found or empty in the app directory.")
            
            # Report Selection
            st.subheader("📑 Select Report to View Details")
            selected_report = st.selectbox(
                "Choose a report:",
                options=list(all_reports_data.keys()),
                format_func=lambda x: f"{x} ({df_comparison[df_comparison['Report Name']==x]['Total Visuals'].values[0]} visuals)"
            )
            
            if selected_report:
                st.divider()
                display_report_data(all_reports_data[selected_report], selected_report)
    else:
        with st.expander("ℹ️ How to use Multiple Files Comparison"):
            st.markdown("""
            ### Steps:
            1. Click on **"Browse files"** above
            2. Select multiple .pbix files (hold Ctrl/Cmd to select multiple)
            3. Wait for all files to be processed
            4. View the comparison summary table
            5. Select individual reports for detailed analysis
            
            ### Comparison Features:
            - Side-by-side metrics for all uploaded reports
            - Quick overview of pages, visuals, fields, and filters
            - Export comparison summary
            - Drill down into any report for full details
            
            ### Use Cases:
            - Compare Dev vs Prod environments
            - Analyze report evolution over time
            - Cross-team report comparison
            - Quality assurance across multiple reports
            """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Power BI Metadata Extractor </small>
</div>
""", unsafe_allow_html=True)

