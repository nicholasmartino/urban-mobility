def get_options(series):
    if series.__class__.__name__ == 'PandasSeries':
        return [{'label': i, 'value': i} for i in series.unique()]

    elif series.__class__.__name__ == 'list':
        return [{'label': i, 'value':i} for i in series]
