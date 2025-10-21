import json
import csv

from src.utils import clean_text, get_type_name

def parse_visual_containers(layout_json, log_file, csv_file_path):
    """
    Parse visual containers from layout JSON and extract columns and measures used in visuals.

    Args:
        layout_json: Parsed JSON object containing layout information
        log_file: File handle to write output
        csv_file_path: Path to CSV file for visual data
    """
    try:
        sections = layout_json.get("sections", [])

        log_file.write(f"\n{'=' * 80}\n")
        log_file.write(f"VISUAL ANALYSIS - COLUMNS AND MEASURES\n")
        log_file.write(f"{'=' * 80}\n\n")

        total_visuals = 0

        # Prepare CSV data
        csv_data = []

        for section_idx, section in enumerate(sections):
            page_name = clean_text(
                section.get("displayName", f"Page {section_idx + 1}")
            )
            visual_containers = section.get("visualContainers", [])

            # Count visuals with projections first
            visuals_with_projections = []
            for visual in visual_containers:
                config_str = visual.get("config", "{}")
                try:
                    config = json.loads(config_str)
                    single_visual = config.get("singleVisual", {})
                    projections = single_visual.get("projections", {})

                    # Check if projections exist and are not empty
                    has_projections = projections and any(projections.values())
                    if has_projections:
                        visuals_with_projections.append(visual)
                except json.JSONDecodeError:
                    continue

            # Only write section header if there are visuals with projections
            if not visuals_with_projections:
                continue

            log_file.write(f"\nPage: {page_name}\n")
            log_file.write(f"{'-' * 80}\n")
            log_file.write(
                f"Total Visuals with Projections: {len(visuals_with_projections)}\n\n"
            )

            for visual_idx, visual in enumerate(visuals_with_projections):
                total_visuals += 1

                # Parse the config JSON string
                config_str = visual.get("config", "{}")
                try:
                    config = json.loads(config_str)
                except json.JSONDecodeError:
                    continue

                single_visual = config.get("singleVisual", {})
                visual_type = clean_text(single_visual.get("visualType", "Unknown"))
                projections = single_visual.get("projections", {})

                log_file.write(f"Visual #{visual_idx + 1}\n")
                log_file.write(f"  Type: {visual_type}\n")

                # Extract columns and measures from projections
                log_file.write(f"  Projections:\n")

                # Collect projection data
                projection_details = []
                for proj_type, proj_items in projections.items():
                    if proj_items:
                        log_file.write(f"    {proj_type}:\n")
                        for item in proj_items:
                            query_ref = clean_text(item.get("queryRef", "N/A"))
                            active = item.get("active", True)
                            log_file.write(f"      - {query_ref} (Active: {active})\n")
                            projection_details.append(
                                {
                                    "projection_type": proj_type,
                                    "query_ref": query_ref,
                                    "active": active,
                                }
                            )

                # Parse dataTransforms if available for additional metadata
                data_transforms_str = visual.get("dataTransforms", "{}")
                field_details = []
                try:
                    data_transforms = json.loads(data_transforms_str)
                    selects = data_transforms.get("selects", [])

                    if selects:
                        log_file.write(f"  Fields Details:\n")
                        for select in selects:
                            display_name = clean_text(select.get("displayName", "N/A"))
                            query_name = clean_text(select.get("queryName", "N/A"))
                            field_type = select.get("type", {})
                            underlying_type = field_type.get("underlyingType", "N/A")
                            format_info = select.get("format", "N/A")

                            # Convert type code to type name
                            if underlying_type != "N/A":
                                type_name = get_type_name(underlying_type)
                            else:
                                type_name = "N/A"

                            # Check if it's a measure (aggregation)
                            expr = select.get("expr", {})
                            aggregation_func = ""
                            is_measure = False
                            if "Aggregation" in expr:
                                agg_func = expr["Aggregation"].get(
                                    "Function", "Unknown"
                                )
                                aggregation_func = f"Function {agg_func}"
                                is_measure = True

                            log_file.write(f"    - Display Name: {display_name}\n")
                            log_file.write(f"      Query Name: {query_name}\n")
                            log_file.write(f"      Type: {type_name}\n")
                            if format_info != "N/A":
                                log_file.write(f"      Format: {format_info}\n")
                            if aggregation_func:
                                log_file.write(
                                    f"      Aggregation: {aggregation_func}\n"
                                )

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

                # Add data to CSV
                # For each field in the visual, create a CSV row
                if field_details:
                    for field in field_details:
                        # Find matching projection
                        matching_projection = None
                        for proj in projection_details:
                            if field["query_name"] in proj["query_ref"]:
                                matching_projection = proj
                                break

                        csv_row = {
                            "Page Name": page_name,
                            "Visual Type": visual_type,
                            "Field Display Name": field["display_name"],
                            "Field Query Name": field["query_name"],
                            "Field Type": field["type"],
                            "Field Format": field["format"]
                            if field["format"] != "N/A"
                            else "",
                            "Is Measure": "Yes" if field["is_measure"] else "No",
                            "Aggregation": field["aggregation"],
                            "Projection Type": matching_projection["projection_type"]
                            if matching_projection
                            else "",
                            "Active": matching_projection["active"]
                            if matching_projection
                            else "",
                        }
                        csv_data.append(csv_row)
                else:
                    # If no field details, add basic visual info
                    for proj in projection_details:
                        csv_row = {
                            "Page Name": page_name,
                            "Visual Type": visual_type,
                            "Field Display Name": "",
                            "Field Query Name": proj["query_ref"],
                            "Field Type": "",
                            "Field Format": "",
                            "Is Measure": "",
                            "Aggregation": "",
                            "Projection Type": proj["projection_type"],
                            "Active": proj["active"],
                        }
                        csv_data.append(csv_row)

                log_file.write(f"\n")

        log_file.write(f"\n{'=' * 80}\n")
        log_file.write(f"SUMMARY\n")
        log_file.write(f"{'=' * 80}\n")
        log_file.write(f"Total Pages: {len(sections)}\n")
        log_file.write(f"Total Visuals with Projections: {total_visuals}\n")
        log_file.write(f"{'=' * 80}\n")

        # Write CSV file
        if csv_data:
            with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
                fieldnames = [
                    "Page Name",
                    "Visual Type",
                    "Field Display Name",
                    "Field Query Name",
                    "Field Type",
                    "Field Format",
                    "Is Measure",
                    "Aggregation",
                    "Projection Type",
                    "Active",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            print(f"CSV file created: {csv_file_path}")
            log_file.write(f"\nCSV file created: {csv_file_path}\n")

    except Exception as e:
        log_file.write(f"\nError parsing visual containers: {str(e)}\n")
