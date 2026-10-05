def validate_parameter(parameter_name, parameter_value, valid_values):
    """Validate parameter value and raise a ValueError with a specific error message.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `str`, `int` or `float`
        Value of the parameter to be validated.
    valid_values : `list`
        List of valid values for the parameter.

    Returns
    -------
    parameter_value : `str`, `int` or `float`
        The validated parameter value.

    Raises
    ------
    ValueError
        If the parameter value is not in the list of valid values.
    """
    # Error message about the invalid parameter
    if parameter_value not in valid_values:
        raise ValueError(
            f"Invalid {parameter_name}: '{parameter_value}'. "
            f"Please use: {_join_options(valid_values)}."
        )

    return parameter_value


def _join_options(values, conjunction="or"):
    """Join values into a human-readable English list.

    Examples
    --------
    >>> _join_options(["a"])
    "'a'"

    >>> _join_options(["a", "b"])
    "'a' or 'b'"

    >>> _join_options(["a", "b"], conjunction="and")
    "'a' and 'b'"

    >>> _join_options(["a", "b", "c"])
    "'a', 'b' or 'c'"
    """
    values = tuple(f"'{value}'" for value in values)

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    return f"{', '.join(values[:-1])} {conjunction} {values[-1]}"

