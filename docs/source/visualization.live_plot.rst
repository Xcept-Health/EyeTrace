.. _visualization.live_plot:

Live Plot
=========

.. automodule:: eyetrace.visualization.live_plot
   :members:
   :undoc-members:
   :show-inheritance:

This submodule contains classes for creating real‑time updating plots using
matplotlib in non‑blocking mode. It is useful for displaying metrics as they
are acquired.

- :class:`LivePlot` : a single‑line plot that updates with each new data point.
- :class:`MultiLivePlot` : multiple subplots sharing a common time axis,
  each displaying a different metric.