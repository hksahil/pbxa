# PBIX Analyser for Reckitt

A comprehensive Power BI (.pbix) file analysis tool built with Streamlit. Extract, analyze, and visualize detailed metadata from Power BI reports including data models, visuals, relationships, and report structure.

## 🌟 Features

### 📊 Comprehensive Analysis
- **Report Summary**: High-level metrics including pages, visuals, slicers, tables, measures, and security settings
- **Page Analysis**: Detailed breakdown of each report page with visual counts and filters
- **Visual Analysis**: Complete inventory of all visuals with their properties, fields, and configurations
- **Data Model Analysis**: Tables, relationships, columns, and statistics
- **DAX Analysis**: Measures, calculated columns, and DAX tables
- **Security Analysis**: Row-level security (RLS) rules
- **M Parameters**: Power Query parameters including incremental refresh detection

### 🔍 Advanced Filtering
- **Global Text Search**: Search across all tables simultaneously
- **Column Filters**: Multi-select filters for each column in every table
- **Smart Highlighting**: Automatically highlights suspicious elements for review
- **Filter Persistence**: Filters maintain state across different views

### 📦 Batch Processing
- **Multi-File Analysis**: Process multiple PBIX files at once
- **Aggregated Results**: Combined analysis across all files
- **Report Comparison**: Easy comparison with "Report Name" column in all tables
- **Progress Tracking**: Visual progress bar for batch operations

### 📥 Export Options
- **Excel Export**: Download all analysis results as a multi-sheet Excel workbook
- **Filtered Data**: Export includes current filter state

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or download this repository**

2. **Create and activate virtual environment**
   ```bash
   cd pbi-extractor-pbixray
   python3 -m venv evn_name
   source evn_name/bin/activate  # On macOS/Linux
   # OR
   evn_name\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Single File Analysis

1. Select "Single File Analysis" mode
2. Upload a PBIX file using the file uploader
3. Wait for analysis to complete
4. Explore different sections:
   - Report Summary (overall metrics)
   - Table Analysis (Power Query tables)
   - Relationships Analysis
   - Column Analysis
   - DAX Measures & Calculated Columns
   - Page Summary
   - Visual Summary
   - Row-Level Security
5. Use the global search to filter across all tables
6. Download the complete analysis as Excel

### Multi-File Analysis

1. Select "Multi-File Analysis" mode
2. Upload multiple PBIX files
3. Click "Start Batch Processing"
4. Wait for all files to be processed
5. View aggregated results with "Report Name" column for easy filtering
6. Use global search to find specific items across all reports
7. Download aggregated analysis as Excel

## 📁 Project Structure

```
pbi-extractor-pbixray/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── FILTERING_FEATURE.md           # Filtering documentation
├── src/
│   ├── constants.py               # Constants and configurations
│   ├── filters.py                 # Filter extraction logic
│   ├── report.py                  # Report metadata extraction
│   ├── utils.py                   # Utility functions
│   └── visuals.py                 # Visual parsing logic
├── pages/
│   └── 2_More_Detailed_Analysis.py # Additional analysis page
└── evn_name/                      # Virtual environment (created during setup)
```

## 🔧 Features by Section

### Report Summary
- Total pages, visuals, slicers (direct/indirect)
- Static elements count
- Filter counts (page + visual level)
- Tables, measures, columns
- Incremental refresh detection
- Row-level security detection

### Visual Summary
- Visual type and title
- Page location
- Field display and query names
- Field types and formats
- Measure detection
- Aggregation functions
- Projection types
- Active state
- **Suspect flag** (highlights visuals that may need review)
- Hidden visual detection

### Table Analysis
- Table names and sources
- Power Query expressions
- Partition information
- Refresh policies

### Relationships Analysis
- From/To tables and columns
- Cardinality (1-to-many, many-to-one, etc.)
- Cross-filter direction
- Active/Inactive state

### Security Analysis
- RLS roles and rules
- Table filters
- DAX filter expressions

## 📦 Dependencies

Core dependencies:
- `streamlit>=1.50.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `openpyxl>=3.1.5` - Excel export
- `lz4>=4.4.4` - Compression support
- `pbixray>=0.1.0` - PBIX extraction library
- `streamlit-aggrid>=0.3.0` - Enhanced data tables

## 🐛 Troubleshooting

### Virtual Environment Issues
If you get "bad interpreter" errors:
```bash
# Remove broken virtual environment
rm -rf evn_name

# Create new one
python3 -m venv evn_name
source evn_name/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PBIX File Not Loading
- Ensure the file is a valid .pbix file
- Check file permissions
- Try re-uploading the file
- Check file size (very large files may take longer)

### Search Not Working
- Clear the search box and try again
- Check for typos in search terms
- Try simpler search terms

## 🔐 Security & Privacy

- All processing is done locally on your machine
- No data is sent to external servers
- PBIX files are processed in-memory
- Session data is cleared when you close the browser

## 📝 Notes

- **Incremental Refresh Detection**: Automatically detects presence of RangeStart parameter
- **Hidden Visuals**: Detects and flags hidden visuals (e.g., for cross-filtering)
- **Slicer Categorization**: Separates direct (visible) and indirect (hidden) slicers
- **Suspect Flagging**: Automatically flags visuals containing specific keywords for review

## 🚧 Known Limitations

1. Very large PBIX files (>500MB) may take significant time to process
2. Some custom visuals may not have complete metadata extraction

## 🛠️ Development

### Adding New Features
1. Create new functions in appropriate modules under `src/`
2. Follow the existing code structure and naming conventions
3. Keep functions focused and reusable
4. Test thoroughly with various PBIX files

### Code Style
- Use clear, descriptive function names
- Add docstrings to all functions
- Follow Python PEP 8 guidelines
- Keep functions focused on single responsibility

## 📄 License

This project is for internal use at Reckitt.

## 👥 Support

For questions, issues, or feature requests:
1. Check the documentation files
2. Review the troubleshooting section
3. Contact the development team

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Application framework
- [PBIXRay](https://github.com/data-goblins/pbixray) - PBIX extraction
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [OpenPyXL](https://openpyxl.readthedocs.io/) - Excel export

---

**Last Updated**: October 2025
