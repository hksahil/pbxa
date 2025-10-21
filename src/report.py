import zipfile
import os
import json
from datetime import datetime

from src.utils import clean_text, get_type_name
from src.filters import extract_filters, format_filters_for_display
from src.visuals import parse_visual_containers, is_visual_hidden


def extract_report_metadata(pbix_file_path):
    """
    Extract comprehensive report metadata including pages, visuals, and filters.

    Args:
        pbix_file_path: Path to the PBIX file or file-like object

    Returns:
        Dictionary containing report metadata
    """
    try:
        report_data = {"summary": {}, "pages": [], "visuals": []}

        # Open PBIX file as ZIP
        with zipfile.ZipFile(pbix_file_path, "r") as zip_ref:
            # Find Layout file
            layout_file_path = None
            for file_path in zip_ref.namelist():
                if file_path == "Report/Layout":
                    layout_file_path = file_path
                    break

            if not layout_file_path:
                return report_data

            # Read and parse Layout file
            with zip_ref.open(layout_file_path) as layout_file:
                layout_bytes = layout_file.read()

                # Try different encodings
                layout_content = None
                encodings_to_try = ["utf-16-le", "utf-16", "utf-8"]

                for encoding in encodings_to_try:
                    try:
                        layout_content = layout_bytes.decode(encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue

                if layout_content is None:
                    layout_content = layout_bytes.decode("utf-8", errors="replace")

                # Parse JSON
                layout_json = json.loads(layout_content)
                sections = layout_json.get("sections", [])

                total_visuals = 0

                # Process each page/section
                for section_idx, section in enumerate(sections):
                    page_name = clean_text(
                        section.get("displayName", f"Page {section_idx + 1}")
                    )
                    visual_containers = section.get("visualContainers", [])

                    # Extract page-level filters (parse JSON string)
                    page_filters_str = section.get("filters", "[]")
                    page_filters = extract_filters(page_filters_str)
                    page_filters_display = format_filters_for_display(page_filters)

                    page_visuals_count = 0
                    page_visuals_with_data = 0
                    page_static_elements = 0
                    page_group_containers = 0
                    page_total_containers = len(visual_containers)

                    # Detect background image at page level (section config)
                    page_background_images = 0
                    section_config_str = section.get("config", "{}")
                    try:
                        section_config = json.loads(section_config_str)
                        objects_cfg = section_config.get("objects", {})
                        background_cfg = objects_cfg.get("background", [])
                        if isinstance(background_cfg, list):
                            for bg in background_cfg:
                                props = bg.get("properties", {})
                                if props.get("image") or props.get("imageUrl"):
                                    page_background_images += 1
                                    break
                    except json.JSONDecodeError:
                        pass

                    # Process each visual
                    for visual_idx, visual in enumerate(visual_containers):
                        config_str = visual.get("config", "{}")
                        try:
                            config = json.loads(config_str)
                            single_visual = config.get("singleVisual", {})
                            projections = single_visual.get("projections", {})
                            visual_type = clean_text(
                                single_visual.get("visualType", "Unknown")
                            )

                            # Count visual groups as static elements (they wrap other visuals)
                            if "singleVisualGroup" in config:
                                page_group_containers += 1
                                page_static_elements += 1
                                continue

                            # Check if projections exist and are not empty
                            has_projections = projections and any(projections.values())

                            # Count static elements (visuals with no projections)
                            if not has_projections:
                                # Check if it's a static element type
                                # Include action buttons and common static visuals
                                static_element_types = [
                                    "image",
                                    "textbox",
                                    "shape",
                                    "button",
                                    "actionbutton",
                                ]
                                if visual_type.lower() in static_element_types:
                                    page_static_elements += 1
                                continue

                            page_visuals_with_data += 1
                            total_visuals += 1

                            visual_id = str(visual.get("id", f"visual_{visual_idx}"))

                            # Extract visual title from vcObjects
                            visual_title = "[No Title]"
                            vc_objects = single_visual.get("vcObjects", {})
                            if vc_objects and "title" in vc_objects:
                                title_settings = vc_objects["title"]
                                if (
                                    isinstance(title_settings, list)
                                    and len(title_settings) > 0
                                ):
                                    for title_obj in title_settings:
                                        properties = title_obj.get("properties", {})
                                        if "text" in properties:
                                            text_expr = properties["text"].get(
                                                "expr", {}
                                            )
                                            if "Literal" in text_expr:
                                                visual_title = clean_text(
                                                    text_expr["Literal"]
                                                    .get("Value", "[No Title]")
                                                    .strip("'")
                                                )
                                                break

                            # Extract visual-level filters (parse JSON string)
                            visual_filters_str = visual.get("filters", "[]")
                            visual_filters = extract_filters(visual_filters_str)
                            visual_filters_display = format_filters_for_display(
                                visual_filters
                            )

                            # Collect projection data
                            projection_details = []
                            for proj_type, proj_items in projections.items():
                                if proj_items:
                                    for item in proj_items:
                                        query_ref = clean_text(
                                            item.get("queryRef", "N/A")
                                        )
                                        active = item.get("active", True)
                                        projection_details.append(
                                            {
                                                "projection_type": proj_type,
                                                "query_ref": query_ref,
                                                "active": active,
                                            }
                                        )

                            # Parse dataTransforms
                            data_transforms_str = visual.get("dataTransforms", "{}")
                            field_details = []
                            try:
                                data_transforms = json.loads(data_transforms_str)
                                selects = data_transforms.get("selects", [])

                                for select in selects:
                                    display_name = clean_text(
                                        select.get("displayName", "N/A")
                                    )
                                    query_name = clean_text(
                                        select.get("queryName", "N/A")
                                    )
                                    field_type = select.get("type", {})
                                    underlying_type = field_type.get(
                                        "underlyingType", "N/A"
                                    )
                                    format_info = select.get("format", "N/A")

                                    # Convert type code to type name
                                    if underlying_type != "N/A":
                                        type_name = get_type_name(underlying_type)
                                    else:
                                        type_name = "N/A"

                                    # Check if it's a measure
                                    expr = select.get("expr", {})
                                    aggregation_func = ""
                                    is_measure = False
                                    if "Aggregation" in expr:
                                        agg_func = expr["Aggregation"].get(
                                            "Function", "Unknown"
                                        )
                                        aggregation_func = f"Function {agg_func}"
                                        is_measure = True

                                    field_details.append(
                                        {
                                            "display_name": display_name,
                                            "query_name": query_name,
                                            "type": type_name,
                                            "format": format_info,
                                            "aggregation": aggregation_func,
                                            "is_measure": is_measure,
                                        }
                                    )

                            except json.JSONDecodeError:
                                pass

                            # Create visual records
                            if field_details:
                                for field in field_details:
                                    # Find matching projection
                                    matching_projection = None
                                    for proj in projection_details:
                                        if field["query_name"] in proj["query_ref"]:
                                            matching_projection = proj
                                            break

                                    visual_row = {
                                        "Page Name": page_name,
                                        "Visual ID": visual_id,
                                        "Visual Title": visual_title,
                                        "Visual Type": visual_type,
                                        "Hidden": "Yes" if is_visual_hidden(visual) else "No",
                                        "Visual Filters": visual_filters_display,
                                        "Field Display Name": field["display_name"],
                                        "Field Query Name": field["query_name"],
                                        "Field Type": field["type"],
                                        "Field Format": field["format"]
                                        if field["format"] != "N/A"
                                        else "",
                                        "Is Measure": "Yes"
                                        if field["is_measure"]
                                        else "No",
                                        "Aggregation": field["aggregation"],
                                        "Projection Type": matching_projection[
                                            "projection_type"
                                        ]
                                        if matching_projection
                                        else "",
                                        "Active": str(matching_projection["active"])
                                        if matching_projection
                                        else "",
                                    }
                                    report_data["visuals"].append(visual_row)
                            else:
                                # If no field details, add basic visual info
                                for proj in projection_details:
                                    visual_row = {
                                        "Page Name": page_name,
                                        "Visual ID": visual_id,
                                        "Visual Title": visual_title,
                                        "Visual Type": visual_type,
                                        "Hidden": "Yes" if is_visual_hidden(visual) else "No",
                                        "Visual Filters": visual_filters_display,
                                        "Field Display Name": "",
                                        "Field Query Name": proj["query_ref"],
                                        "Field Type": "",
                                        "Field Format": "",
                                        "Is Measure": "",
                                        "Aggregation": "",
                                        "Projection Type": proj["projection_type"],
                                        "Active": str(proj["active"]),
                                    }
                                    report_data["visuals"].append(visual_row)

                        except json.JSONDecodeError:
                            continue

                    # Add page summary
                    page_data = {
                        "Page Name": page_name,
                        "Visual Count": page_visuals_with_data,
                        "Static Elements": page_static_elements + page_background_images,
                        "All Elements": page_total_containers + page_background_images,
                        "Groups": page_group_containers,
                        "Page Filters": page_filters_display
                        if page_filters_display
                        else "None",
                    }
                    report_data["pages"].append(page_data)

                # Calculate total static elements
                total_static_elements = sum(page.get("Static Elements", 0) for page in report_data["pages"])

                # Add report summary
                report_data["summary"] = {
                    "Total Pages": len(sections),
                    "Total Visuals": total_visuals,
                    "Total Static Elements": total_static_elements,
                }

        return report_data

    except Exception as e:
        raise Exception(f"Error extracting report metadata: {str(e)}")


def extract_pbix_contents(
    pbix_file_path, output_log_path="output.log", output_csv_path="visuals_data.csv"
):
    """
    Reads a PBIX file as a ZIP archive and lists all files with their paths.
    Writes the output to a log file.

    Args:
        pbix_file_path: Path to the PBIX file
        output_log_path: Path to the output log file (default: output.log)
        output_csv_path: Path to the output CSV file (default: visuals_data.csv)
    """
    try:
        # Check if PBIX file exists
        if not os.path.exists(pbix_file_path):
            print(f"Error: PBIX file not found at {pbix_file_path}")
            return

        # Open log file for writing
        with open(output_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"PBIX File Analysis Report\n")
            log_file.write(f"{'=' * 80}\n")
            log_file.write(f"File: {pbix_file_path}\n")
            log_file.write(
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_file.write(f"{'=' * 80}\n\n")

            # Open PBIX file as ZIP
            with zipfile.ZipFile(pbix_file_path, "r") as zip_ref:
                # Get list of all files in the archive
                file_list = zip_ref.namelist()

                log_file.write(f"Total files found: {len(file_list)}\n\n")
                log_file.write(f"Files and Folders:\n")
                log_file.write(f"{'-' * 80}\n")

                # Separate folders and files
                folders = set()
                files = []
                layout_file_path = None

                for file_path in file_list:
                    # Check if it's a directory (ends with /)
                    if file_path.endswith("/"):
                        folders.add(file_path)
                        log_file.write(f"[FOLDER] {file_path}\n")
                    else:
                        files.append(file_path)
                        # Extract folder path from file path
                        folder_path = os.path.dirname(file_path)
                        if folder_path:
                            folders.add(folder_path + "/")

                        # Get file info
                        file_info = zip_ref.getinfo(file_path)
                        file_size = file_info.file_size

                        log_file.write(
                            f"[FILE]   {file_path} (Size: {file_size:,} bytes)\n"
                        )

                        # Check if this is the Layout file
                        if file_path == "Report/Layout":
                            layout_file_path = file_path

                # Summary
                log_file.write(f"\n{'=' * 80}\n")
                log_file.write(f"Summary:\n")
                log_file.write(f"  Total Folders: {len(folders)}\n")
                log_file.write(f"  Total Files: {len(files)}\n")
                log_file.write(f"{'=' * 80}\n\n")

                # Extract and write Layout file contents
                layout_json = None
                if layout_file_path:
                    log_file.write(f"\n{'=' * 80}\n")
                    log_file.write(f"LAYOUT FILE CONTENTS (Report/Layout)\n")
                    log_file.write(f"{'=' * 80}\n\n")

                    try:
                        # Read the Layout file content
                        with zip_ref.open(layout_file_path) as layout_file:
                            layout_bytes = layout_file.read()

                            # Try different encodings
                            layout_content = None
                            encodings_to_try = ["utf-16-le", "utf-16", "utf-8"]

                            for encoding in encodings_to_try:
                                try:
                                    layout_content = layout_bytes.decode(encoding)
                                    # If successful, break
                                    break
                                except (UnicodeDecodeError, UnicodeError):
                                    continue

                            if layout_content is None:
                                # Fallback to utf-8 with error handling
                                layout_content = layout_bytes.decode(
                                    "utf-8", errors="replace"
                                )

                            # Try to parse as JSON and pretty print
                            try:
                                layout_json = json.loads(layout_content)
                                formatted_json = json.dumps(
                                    layout_json, indent=2, ensure_ascii=False
                                )
                                log_file.write(formatted_json)
                            except json.JSONDecodeError:
                                # If not valid JSON, write as-is
                                log_file.write(layout_content)

                        log_file.write(f"\n\n{'=' * 80}\n")
                        log_file.write(f"END OF LAYOUT FILE\n")
                        log_file.write(f"{'=' * 80}\n")

                    except Exception as e:
                        log_file.write(f"Error reading Layout file: {str(e)}\n")
                else:
                    log_file.write(f"\nNote: Layout file not found in PBIX archive.\n")

                # Parse visual containers if layout JSON was successfully loaded
                if layout_json:
                    parse_visual_containers(layout_json, log_file, output_csv_path)

        print(f"Analysis complete! Output written to: {output_log_path}")

    except zipfile.BadZipFile:
        print(f"Error: {pbix_file_path} is not a valid ZIP/PBIX file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Specify your PBIX file path
    pbix_path = "./data/Components requirements next 12 weeks by ORDERING plant.pbix"
    output_log = "output.log"
    output_csv = "visuals_data.csv"

    extract_pbix_contents(pbix_path, output_log, output_csv)
