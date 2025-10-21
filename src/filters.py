import json
from src.utils import clean_text


def parse_filter_item(filter_item):
    """
    Parse individual filter item to extract meaningful information.

    Args:
        filter_item: Filter dictionary

    Returns:
        Dictionary with filter details or None if no meaningful filter
    """
    try:
        # Extract filter conditions - only process if Where clause exists
        filter_data = filter_item.get("filter", {})
        where_clause = filter_data.get("Where", [])

        # Skip filters without Where clause
        if not where_clause:
            return None

        # Extract basic filter info
        filter_type = filter_item.get("type", "Unknown")
        expression = filter_item.get("expression", {})

        # Extract table and column from expression
        table_name = "Unknown"
        column_name = "Unknown"

        if "Column" in expression:
            column_info = expression["Column"]

            # Get table name from Entity
            entity_expr = column_info.get("Expression", {})
            if "SourceRef" in entity_expr:
                source_ref = entity_expr["SourceRef"]
                table_name = source_ref.get("Entity", "Unknown")

            # Get column name from Property
            column_name = column_info.get("Property", "Unknown")

        # Helper function to clean Power BI literal values
        def clean_literal_value(value_str):
            """Remove Power BI type suffixes from literal values"""
            if not isinstance(value_str, str):
                return str(value_str)

            # Remove common Power BI type suffixes
            # D = Decimal, L = Long, M = Money/Currency
            if (
                value_str.endswith("D")
                or value_str.endswith("L")
                or value_str.endswith("M")
            ):
                # Check if it's actually a number with suffix
                base_value = value_str[:-1]
                # Check if base is numeric (including decimals and negatives)
                if (
                    base_value.replace("-", "")
                    .replace(".", "")
                    .replace(",", "")
                    .isdigit()
                ):
                    return base_value

            return value_str

        # Extract filter conditions
        conditions = []
        operator = "Unknown"

        for condition in where_clause:
            if "Condition" in condition:
                cond = condition["Condition"]

                # Handle "In" operator
                if "In" in cond:
                    in_clause = cond["In"]
                    values = in_clause.get("Values", [])

                    if values:
                        operator = "IN"
                        # Extract literal values
                        extracted_values = []
                        for value_array in values:
                            if isinstance(value_array, list):
                                for val in value_array:
                                    if isinstance(val, dict) and "Literal" in val:
                                        literal_val = val["Literal"].get("Value", "")
                                        cleaned_val = clean_literal_value(literal_val)
                                        extracted_values.append(cleaned_val)

                        if extracted_values:
                            # Show all values without truncation
                            conditions.append(f"IN ({', '.join(extracted_values)})")
                        else:
                            conditions.append(f"IN ({len(values)} values)")

                # Handle "Not" operator with "In"
                elif "Not" in cond:
                    not_clause = cond["Not"]
                    if "Expression" in not_clause and "In" in not_clause["Expression"]:
                        in_clause = not_clause["Expression"]["In"]
                        values = in_clause.get("Values", [])
                        operator = "NOT IN"

                        if values:
                            extracted_values = []
                            for value_array in values:
                                if isinstance(value_array, list):
                                    for val in value_array:
                                        if isinstance(val, dict) and "Literal" in val:
                                            literal_val = val["Literal"].get(
                                                "Value", ""
                                            )
                                            cleaned_val = clean_literal_value(
                                                literal_val
                                            )
                                            extracted_values.append(cleaned_val)

                            if extracted_values:
                                # Show all values without truncation
                                conditions.append(
                                    f"NOT IN ({', '.join(extracted_values)})"
                                )
                            else:
                                conditions.append(f"NOT IN ({len(values)} values)")

                # Handle comparison operators
                elif "Comparison" in cond:
                    comparison = cond["Comparison"]
                    comparison_kind = comparison.get("ComparisonKind", 0)

                    # Map comparison kind to operator
                    comparison_map = {0: "=", 1: "<>", 2: ">", 3: ">=", 4: "<", 5: "<="}
                    operator = comparison_map.get(comparison_kind, "Unknown")

                    # Try to get the comparison value
                    right_expr = comparison.get("Right", {})
                    if "Literal" in right_expr:
                        value = right_expr["Literal"].get("Value", "")
                        cleaned_value = clean_literal_value(value)
                        conditions.append(f"{operator} {cleaned_value}")
                    else:
                        conditions.append(operator)

                # Handle "Between" operator
                elif "Between" in cond:
                    operator = "BETWEEN"
                    between_clause = cond["Between"]
                    lower = between_clause.get("Lower", {})
                    upper = between_clause.get("Upper", {})

                    lower_val = ""
                    upper_val = ""

                    if "Literal" in lower:
                        lower_val = clean_literal_value(
                            lower["Literal"].get("Value", "")
                        )
                    if "Literal" in upper:
                        upper_val = clean_literal_value(
                            upper["Literal"].get("Value", "")
                        )

                    if lower_val and upper_val:
                        conditions.append(f"BETWEEN {lower_val} AND {upper_val}")
                    else:
                        conditions.append("BETWEEN (values)")

                # Handle "Contains" operator
                elif "Contains" in cond:
                    operator = "CONTAINS"
                    contains_clause = cond["Contains"]
                    right_expr = contains_clause.get("Right", {})

                    if "Literal" in right_expr:
                        value = right_expr["Literal"].get("Value", "")
                        cleaned_value = clean_literal_value(value)
                        conditions.append(f"CONTAINS '{cleaned_value}'")
                    else:
                        conditions.append("CONTAINS (text)")

        # Only return filter if we have meaningful conditions
        if not conditions:
            return None

        return {
            "table": clean_text(table_name),
            "column": clean_text(column_name),
            "type": filter_type,
            "operator": operator,
            "conditions": conditions,
        }

    except Exception as e:
        return None


def extract_filters(filters_json):
    """
    Extract and format filter information from filters JSON.

    Args:
        filters_json: Filters JSON string or object

    Returns:
        List of filter dictionaries with detailed information
    """
    filter_list = []

    try:
        # Parse JSON if string
        if isinstance(filters_json, str):
            if not filters_json or filters_json.strip() == "":
                return []
            filters = json.loads(filters_json)
        else:
            filters = filters_json

        if not filters:
            return []

        # Handle different filter structures
        if isinstance(filters, list):
            for filter_item in filters:
                filter_info = parse_filter_item(filter_item)
                if filter_info:
                    filter_list.append(filter_info)
        elif isinstance(filters, dict):
            filter_info = parse_filter_item(filters)
            if filter_info:
                filter_list.append(filter_info)

    except (json.JSONDecodeError, Exception) as e:
        pass

    return filter_list


def format_filters_for_display(filters):
    """
    Format filter list for display in UI.

    Args:
        filters: List of filter dictionaries

    Returns:
        Formatted string for display
    """
    if not filters:
        return ""

    formatted_filters = []
    for f in filters:
        table = f.get("table", "Unknown")
        column = f.get("column", "Unknown")
        conditions = f.get("conditions", [])

        if conditions:
            condition_str = " ".join(conditions)
            formatted_filters.append(f"{table}.{column}: {condition_str}")
        else:
            formatted_filters.append(f"{table}.{column}")

    return " | ".join(formatted_filters)

