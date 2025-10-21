# Text Filtering Feature

## Overview
A global text filtering feature has been added to the PBIX Analyser application. This allows users to filter all rows across all tables/dataframes based on text input.

## Changes Made

### 1. New Utility Function (`src/utils.py`)
Added a new function `filter_dataframe_by_text()` that:
- Accepts a pandas DataFrame and search text as parameters
- Searches across ALL columns in the dataframe (case-insensitive)
- Returns only rows that contain the search text in any column
- Handles edge cases (empty dataframes, empty search text, etc.)

### 2. Main App Page (`app.py`)
Added text filtering to the main PBIX analysis page:
- **Global Search Input**: Added BELOW the Report Summary section
- **Report Summary NOT Filtered**: The Report Summary matrix is excluded from filtering
- **Applied to Tables Below**: The filter is applied to the following 9 sections:
  1. Table Analysis
  2. Relationships Analysis
  3. Column Analysis
  4. M Parameters
  5. DAX Tables
  6. Calculated Columns
  7. DAX Measures
  8. Page Summary
  9. Visual Summary
- **Visual Feedback**: Shows "Filtering all tables below for: 'search_text'" when active
- **Result Count**: Displays "Showing X of Y rows" for each filtered table

### 3. Detailed Analysis Page (`pages/2_More_Detailed_Analysis.py`)
Added text filtering to the detailed analysis page:
- **Global Search Input**: Added at the top of each report view
- **Applied Across All Tabs**:
  - Tab 1 (Page Overview): Filters pages summary
  - Tab 2 (Visual Details): Filters visual records (in addition to existing filters)
  - Tab 3 (Filters): Filters the filters table
  - Tab 4 (Export Data): Export shows filtered data
- **Works with Existing Filters**: Text filter works in combination with existing dropdown/multiselect filters
- **Result Count Feedback**: Shows before/after counts when text filter reduces results

## How to Use

### For Users:
1. **Upload a PBIX file** in the app
2. **Enter search text** in the "Search across all tables/dataframes" input field
3. **All tables automatically filter** to show only rows containing the search text
4. **Clear the input** to see all data again

### Example Use Cases:
- Search for a specific table name: "Sales"
- Search for a specific measure: "Total Revenue"
- Search for a specific page: "Overview"
- Search for a specific field: "CustomerName"
- Search for keywords: "eh", "bu", "functional"

## Technical Details

### Function Signature:
```python
def filter_dataframe_by_text(df, search_text):
    """
    Filter a DataFrame by searching for text across all columns.
    
    Args:
        df: pandas DataFrame to filter
        search_text: Text to search for (case-insensitive)
    
    Returns:
        Filtered DataFrame containing only rows that match the search text
    """
```

### Key Features:
- **Case-insensitive search**: "sales" will match "Sales", "SALES", etc.
- **Searches all columns**: Automatically checks every column in the dataframe
- **Exact substring match**: Searches for the text as a substring (not regex)
- **Error handling**: Gracefully skips columns that can't be searched
- **Performance**: Efficient pandas operations using boolean masking

## Benefits

1. **Quick Data Discovery**: Find specific information across multiple tables instantly
2. **Better User Experience**: No need to scroll through large tables
3. **Combines with Existing Filters**: Works alongside dropdown and multiselect filters
4. **Consistent Across App**: Same filtering behavior on all pages
5. **Non-destructive**: Original data remains intact; only display is filtered

## Future Enhancements (Optional)

Consider these potential improvements:
- Add regex search option
- Add column-specific search
- Add case-sensitive search toggle
- Add export filtered results option
- Add search history
- Add highlight matching text in results

