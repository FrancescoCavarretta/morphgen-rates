import pandas as pd
from pathlib import Path


def _local_data_path(filename='morph_data', ext="csv"):
  """
  Build a path like: <this_file_dir>/data/<filename>.<ext>

  Parameters
  ----------
  filename : str
      Base filename (without extension)
  ext : str, default "csv"
      File extension (without the dot)

  Returns
  -------
  pathlib.Path
      Full path to the data file
  """
  work_dir = Path(__file__).resolve().parent
  return work_dir / f"{filename}.{ext}"


def get_data(area, neuron_type):
  """
  Retrieve summary morphology statistics for a given brain area and neuron class.

  This function loads a local CSV dataset, filters rows matching the requested
  `area` and `neuron_type`, and aggregates statistics by `section_type`. The
  output is a nested dictionary keyed by section type (e.g., soma, apical, basal),
  containing:

  - Summary statistics for bifurcation counts and total length
  - Estimated number of primary neurites at the soma (Count0)
  - Sholl plot summary statistics (bin size, mean counts, standard deviation)

  Parameters
  ----------
  area : str
      Brain region identifier used in the dataset (must match values in the
      'area' column of the CSV)
  neuron_type : str
      Neuron class identifier used in the dataset (must match values in the
      'neuron_type' column of the CSV)

  Returns
  -------
  dict
      Nested dictionary structured as:

      data = {
        "<section_type>": {
          "bifurcation_count": {"mean": ..., "std": ..., "min": ..., "max": ...},
          "total_length": {"mean": ..., "std": ..., "min": ..., "max": ...},
          "primary_count": {"mean": ..., "std": ..., "min": ..., "max": ...},
          "sholl_plot": {
            "bin_size": float,
            "mean": list[float],
            "std": list[float],
          },
        },
        ...
      }

      Notes on fields:
      - `primary_count` corresponds to the row group labeled 'Count0'
      - Sholl values are collected from rows whose metric name starts with 'Count'
        (including 'Count0'); users may want to interpret/plot them as a function
        of radial bin index multiplied by `bin_size`

  Raises
  ------
  AssertionError
      If no rows match the requested `area` and `neuron_type`

  Notes
  -----
  - The function expects the local CSV to include at least the following columns:
      'area', 'neuron_type', 'neuron_name', 'section_type', 'bin_size'
      plus metric columns including:
        - 'bifurcation_count'
        - 'total_length'
        - 'Count0', 'Count1', ... (Sholl counts per radial bin)
  - Statistics are computed using `pandas.DataFrame.groupby(...).describe()`.
    Only the summary columns 'mean', 'std', 'min', 'max' are retained.

  Examples
  --------
  >>> data = get_data("CTX", "pyr")
  >>> data["apical"]["bifurcation_count"]["mean"]
  42.0
  >>> data["apical"]["sholl_plot"]["bin_size"]
  50.0
  >>> len(data["apical"]["sholl_plot"]["mean"])
  20
  """
  
  data = {}
    
  # load data
  df = pd.read_csv(_local_data_path(), index_col=0)

  # select specific area and neuron type
  df = df[(df['area'] == area) & (df['neuron_type'] == neuron_type)]

  # ensure that there are area and neuron_type in the df
  assert df.shape[0] > 0, "The area {area} or neuron class {neuron_type} are not known"
  
  # neuron name unnecessary
  df.drop(['area', 'neuron_type', 'neuron_name'], axis=1, inplace=True)

  # statistics
  df = df.groupby('section_type').describe()

  # select only a subset of columns
  df = df.loc[:, df.columns.get_level_values(1).isin(['mean', 'std', 'min', 'max'])]

  # get subsections
  for section_type, row in df.iterrows():
    data[section_type] = {}

    # get statistics
    for data_type in ['bifurcation_count', 'total_length']:
      tmp = row.loc[row.index.get_level_values(0) == data_type, :]
      tmp.index = tmp.index.droplevel(0)
      data[section_type][data_type] = tmp.to_dict()

    # count neurites at the soma
    tmp = row.loc[row.index.get_level_values(0) == 'Count0', :]
    tmp.index = tmp.index.droplevel(0)
    data[section_type]['primary_count'] = tmp.to_dict()
    
    # sholl plots
    tmp = row.loc[row.index.get_level_values(0).str.startswith('Count'), :]
    data[section_type]['sholl_plot'] = {
      'bin_size':row[('bin_size', 'mean')].tolist(),
      'mean':tmp.loc[tmp.index.get_level_values(1) == 'mean', :].tolist(),
      'std':tmp.loc[tmp.index.get_level_values(1) == 'std', :].tolist()
      }

  return data

