"""
Centralized logic for building InfluxDB Flux queries.
"""

def _build_flux_query(bucket, measurement, fields, range_start, range_stop=None, 
                      scale=1.0, offset='0m', interpolate=False, downsample=False, verbose=True):
    """
    Build a standard Flux query with scaling, time offset, and optional interpolation.
    
    Args:
        bucket: InfluxDB bucket name
        measurement: Measurement name
        fields: Field name (str) or list of field names
        range_start: Time range start (e.g., '-30d' or 'date.sub(d: 48h, from: now())')
        range_stop: Time range stop (e.g., 'now()' or '14d'), default None
        scale: Scaling factor to apply
        offset: Time offset to apply

        interpolate: If True, use linear interpolation to fill gaps at 15min intervals
        downsample: If True, aggregate to 15min windows before processing (for high-freq data)
        verbose: If True, print the generated query
    
    Returns:
        str: Flux query string
    """
    # Handle single field or list of fields
    if isinstance(fields, str):
        field_filter = f'r["_field"] == "{fields}"'
    else:
        field_filter = " or ".join([f'r["_field"] == "{f}"' for f in fields])
    
    # Build range clause
    if range_stop:
        range_clause = f'range(start: {range_start}, stop: {range_stop})'
    elif interpolate:
        range_clause = f'range(start: {range_start}, stop: now())'
    else:
        range_clause = f'range(start: {range_start})'
    
    # Check if we need imports for dynamic time expressions or interpolation
    needs_date_import = 'date.sub' in range_start or (range_stop and 'date.' in range_stop)
    import_stmt = ''
    if needs_date_import:
        import_stmt += 'import "date"\n    '
    if interpolate:
        import_stmt += 'import "interpolate"\n    '
    
    query = f'''
    {import_stmt}from(bucket: "{bucket}")
      |> {range_clause}
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => {field_filter})
    '''
    
    if scale != 1.0:
        query += f'  |> map(fn: (r) => ({{ r with _value: r._value * {scale} }}))\n'
    
    # Downsample high-frequency data (e.g. 10s from live bucket) to 15min
    if downsample:
        query += '  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)\n'
        
    # Linear interpolation to fill gaps
    if interpolate:
        query += '  |> interpolate.linear(every: 15m)\n'
    
    if offset != '0m':
        query += f'  |> timeShift(duration: {offset})\n'
    
    query += '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
    query += '  |> unique(column: "_time")\n'
    
    if verbose:
        print(f"[FLUX QUERY]\n{query.strip()}")
    
    return query
